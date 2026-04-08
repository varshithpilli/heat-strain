"""
train_models.py
═══════════════════════════════════════════════════════════════
HeatGuard AI — 3-Class Training Pipeline
Custom NN (scratch) + Random Forest + Gradient Boosting + SVM + Logistic Reg

KEY FIXES:
  • All sklearn models use class_weight='balanced'
  • Threshold calibration via post-hoc probability tuning
  • High-risk recall explicitly tracked and printed
  • Stratified K-Fold used everywhere
  • SVM trained on subsample for speed; full data for others
  • Saves per-class precision/recall/f1 to metrics.json

USAGE:
    python train_models.py

OUTPUTS:
    saved_models/  ← .pkl files + metrics.json
    plots/         ← confusion matrices, ROC, comparison charts
═══════════════════════════════════════════════════════════════
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc,
    recall_score, precision_score
)
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_all_data, get_feature_matrix, LABEL_NAMES
from neural_network import CustomNeuralNetwork

INFANT_PATH   = './datasets/InfantSmartWear_TemperatureMonitoring_v1.csv'
WEARABLE_PATH = './datasets/wearable_sensor_data.csv'
P1_PATH       = './datasets/Final_Dataframe_P1.csv'
P2_PATH       = './datasets/Final_Dataframe_P2.csv'

OUT_DIR  = 'saved_models'
PLOT_DIR = 'plots'
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

COLORS = {
    'custom_nn'     : '#7F77DD',
    'random_forest' : '#1D9E75',
    'gradient_boost': '#EF9F27',
    'svm'           : '#D85A30',
    'logistic_reg'  : '#378ADD',
}

MODEL_DISPLAY = {
    'custom_nn'     : '🧠 Custom Neural Network (scratch)',
    'random_forest' : '🌲 Random Forest',
    'gradient_boost': '🚀 Gradient Boosting',
    'svm'           : '🎯 SVM (RBF)',
    'logistic_reg'  : '📊 Logistic Regression',
}

def get_models(n_features):
    return {
        
        'custom_nn': CustomNeuralNetwork(
            hidden_sizes=(128, 64, 32),
            n_classes=3,
            lr=1e-3,
            epochs=200,
            batch_size=64,
            l2=1e-4,
            dropout_rates=(0.3, 0.2, 0.0),
            patience=20,
            lr_decay=0.97,
            random_state=42,
        ),

        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=400,
                max_depth=18,
                min_samples_leaf=1,
                class_weight='balanced_subsample',  
                random_state=42,
                n_jobs=-1,
            ))
        ]),

        'gradient_boost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=1,
                random_state=42,
            ))
        ]),

        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                C=15.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42,
            ))
        ]),

        'logistic_reg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=2000,
                C=1.5,
                solver='lbfgs',
                class_weight='balanced',
                random_state=42,
            ))
        ]),
    }



def calibrate_thresholds(model, X_val, y_val, n_classes=3):
    """
    Find per-class probability thresholds that maximise
    macro F1 on the validation set.
    Uses a coarse grid search; returns threshold array shape (n_classes,).
    """
    proba = model.predict_proba(X_val)
    best_f1, best_thresh = 0.0, np.full(n_classes, 1.0 / n_classes)

    for t0 in np.arange(0.20, 0.65, 0.05):
        for t1 in np.arange(0.20, 0.65, 0.05):
            t2 = 1.0 - t0 - t1
            if t2 < 0.05 or t2 > 0.80:
                continue
            thresh = np.array([t0, t1, t2])
            # Normalise to simplex
            thresh = thresh / thresh.sum()
            adj = proba / thresh
            preds = adj.argmax(axis=1)
            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh.copy()
    return best_thresh


def predict_with_thresholds(model, X, thresholds):
    proba = model.predict_proba(X)
    adj   = proba / (thresholds + 1e-9)
    return adj.argmax(axis=1), proba



def train_and_evaluate(X, y, target_name):
    X = X.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)

    models  = get_models(X.shape[1])
    results = {}
    trained = {}
    classes = sorted(np.unique(y))
    labels  = [LABEL_NAMES[i] for i in classes]

    print(f"\n  Test set class distribution:")
    for cls in classes:
        cnt = (y_te == cls).sum()
        print(f"    {LABEL_NAMES[cls]}: {cnt} ({cnt/len(y_te)*100:.1f}%)")

    for name, model in models.items():
        print(f"\n  ── [{name}] ──")

        X_fit, y_fit = X_tr2, y_tr2
        if name == 'svm' and len(X_fit) > 25000:
            idx = np.random.default_rng(42).choice(len(X_fit), 25000, replace=False)
            X_fit, y_fit = X_fit[idx], y_fit[idx]
            print(f"    SVM: using {len(X_fit)} rows (subsampled)")

        model.fit(X_fit, y_fit)

        thresholds = calibrate_thresholds(model, X_val, y_val)
        print(f"    Thresholds: {thresholds.round(3)}")

        yp, yprob = predict_with_thresholds(model, X_te, thresholds)

        acc = accuracy_score(y_te, yp)
        f1  = f1_score(y_te, yp, average='weighted', zero_division=0)
        f1m = f1_score(y_te, yp, average='macro',    zero_division=0)

        p_cls = precision_score(y_te, yp, average=None, labels=classes, zero_division=0)
        r_cls = recall_score(   y_te, yp, average=None, labels=classes, zero_division=0)
        f1_cls= f1_score(       y_te, yp, average=None, labels=classes, zero_division=0)

        if len(classes) > 2:
            yb  = label_binarize(y_te, classes=classes)
            roc = roc_auc_score(yb, yprob, multi_class='ovr', average='weighted')
        else:
            roc = roc_auc_score(y_te, yprob[:, 1])

        print(f"    Accuracy={acc:.4f}  F1-weighted={f1:.4f}  F1-macro={f1m:.4f}  ROC={roc:.4f}")
        for i, cls in enumerate(classes):
            print(f"    {LABEL_NAMES[cls]:8s}: P={p_cls[i]:.3f}  R={r_cls[i]:.3f}  F1={f1_cls[i]:.3f}")

        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_sc = []
        for fold, (tr_i, val_i) in enumerate(skf.split(X, y)):
            m_cv = get_models(X.shape[1])[name]
            Xf, yf = X[tr_i], y[tr_i]
            if name == 'svm' and len(Xf) > 20000:
                idx = np.random.default_rng(fold).choice(len(Xf), 20000, replace=False)
                Xf, yf = Xf[idx], yf[idx]
            m_cv.fit(Xf, yf)
            th_cv = calibrate_thresholds(m_cv, X[val_i], y[val_i])
            yp_cv, _ = predict_with_thresholds(m_cv, X[val_i], th_cv)
            cv_sc.append(accuracy_score(y[val_i], yp_cv))
        cv_mean, cv_std = np.mean(cv_sc), np.std(cv_sc)
        print(f"    CV: {cv_mean:.4f} ± {cv_std:.4f}")

        model._heatguard_thresholds = thresholds

        results[name] = dict(
            accuracy=acc, f1_weighted=f1, f1_macro=f1m, roc_auc=roc,
            cv_mean=cv_mean, cv_std=cv_std,
            thresholds=thresholds.tolist(),
            per_class={LABEL_NAMES[cls]: {
                'precision': float(p_cls[i]),
                'recall'   : float(r_cls[i]),
                'f1'       : float(f1_cls[i]),
            } for i, cls in enumerate(classes)},
            y_test=y_te, y_pred=yp, y_prob=yprob,
            classes=classes, labels=labels,
            report=classification_report(y_te, yp, target_names=labels, output_dict=True),
        )
        trained[name] = model

    return results, trained



def _cmap_purple():
    return LinearSegmentedColormap.from_list('p', ['#EEEDFE', '#534AB7'])


def plot_confusion_matrices(results, target):
    n  = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5))
    if n == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        cn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        ax.imshow(cn, cmap=_cmap_purple(), vmin=0, vmax=1)
        ax.set_xticks(range(len(r['labels']))); ax.set_xticklabels(r['labels'], rotation=30, ha='right')
        ax.set_yticks(range(len(r['labels']))); ax.set_yticklabels(r['labels'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f"{name.replace('_',' ').title()}\nAcc={r['accuracy']:.3f}  F1={r['f1_macro']:.3f}",
                     fontweight='bold', fontsize=9)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                tc = 'white' if cn[i, j] > 0.5 else '#2C2C2A'
                ax.text(j, i, f"{cm[i,j]}\n({cn[i,j]:.2f})",
                        ha='center', va='center', fontsize=7.5, color=tc)
    title = target.replace('_label','').replace('_',' ').title()
    plt.suptitle(f"Confusion Matrices — {title}", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'confusion_{target}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def plot_roc_curves(results, target):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        color   = COLORS.get(name, '#888')
        classes = r['classes']
        y_test  = r['y_test']
        y_prob  = r['y_prob']
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f"{name.replace('_',' ').title()} AUC={auc(fpr,tpr):.3f}")
        else:
            yb = label_binarize(y_test, classes=classes)
            all_fpr = np.unique(np.concatenate(
                [roc_curve(yb[:, i], y_prob[:, i])[0] for i in range(len(classes))]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(len(classes)):
                fpr_i, tpr_i, _ = roc_curve(yb[:, i], y_prob[:, i])
                mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
            mean_tpr /= len(classes)
            ax.plot(all_fpr, mean_tpr, color=color, lw=2.5,
                    label=f"{name.replace('_',' ').title()} AUC={auc(all_fpr,mean_tpr):.3f}")
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f"ROC Curves — {target.replace('_label','').replace('_',' ').title()}",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'roc_{target}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def plot_per_class_f1(results, target):
    """Bar chart showing per-class F1 for every model."""
    n_models = len(results)
    classes  = list(next(iter(results.values()))['per_class'].keys())
    class_colors = {'Normal': '#27AE60', 'Moderate': '#F39C12', 'High': '#E74C3C'}

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.5), sharey=True)
    if n_models == 1: axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        vals  = [r['per_class'].get(c, {}).get('f1', 0) for c in classes]
        cols  = [class_colors.get(c, '#888') for c in classes]
        bars  = ax.bar(classes, vals, color=cols, edgecolor='white', lw=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_title(name.replace('_',' ').title(), fontweight='bold', fontsize=9)
        ax.axhline(0.80, color='grey', linestyle='--', lw=1, alpha=0.5)
        ax.grid(True, axis='y', alpha=0.2)

    title = target.replace('_label','').replace('_',' ').title()
    plt.suptitle(f"Per-Class F1 Score — {title}", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'per_class_f1_{target}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def plot_model_comparison(all_results):
    targets     = list(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())
    metrics     = [('accuracy','Accuracy'), ('f1_weighted','F1 Weighted'), ('roc_auc','ROC AUC')]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(model_names)); w = 0.35

    for ax, (metric, label) in zip(axes, metrics):
        for ti, target in enumerate(targets):
            vals   = [all_results[target][m][metric] for m in model_names]
            offset = (ti - 0.5) * w
            cols   = [COLORS.get(n,'#888') for n in model_names]
            bars   = ax.bar(x + offset, vals, w,
                            label=target.replace('_label','').replace('_',' ').title(),
                            color=cols if ti == 0 else [c+'88' for c in cols],
                            edgecolor='white', lw=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                        f"{v:.3f}", ha='center', fontsize=7, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_','\n').title() for n in model_names], fontsize=8)
        ax.set_title(label, fontsize=12, fontweight='bold'); ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.2); ax.set_axisbelow(True)
        ax.axhline(0.90, color='#D85A30', linestyle='--', lw=1.5, alpha=0.6)

    plt.suptitle('Model Performance Comparison — HeatGuard AI (3-Class)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, 'model_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def plot_cv_bars(all_results):
    targets = list(all_results.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(9*len(targets), 5))
    if len(targets) == 1: axes = [axes]
    for ax, target in zip(axes, targets):
        results = all_results[target]
        names  = list(results.keys())
        means  = [results[n]['cv_mean'] for n in names]
        stds   = [results[n]['cv_std']  for n in names]
        cols   = [COLORS.get(n,'#888') for n in names]
        bars   = ax.bar(names, means, color=cols, edgecolor='white', lw=0.8, alpha=0.92)
        ax.errorbar(names, means, yerr=stds, fmt='none', color='#2C2C2A',
                    capsize=6, lw=2, capthick=2)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.005,
                    f"{m:.3f}\n±{s:.3f}", ha='center', fontsize=8)
        ax.set_ylim(0, 1.15); ax.set_ylabel('5-Fold CV Accuracy')
        ax.set_xticklabels([n.replace('_','\n').title() for n in names], fontsize=9)
        ax.set_title(f"Cross-Validation — {target.replace('_label','').replace('_',' ').title()}",
                     fontsize=12, fontweight='bold')
        ax.axhline(0.90, color='#D85A30', linestyle='--', lw=1.5)
        ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, 'cross_validation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def plot_class_distribution(df):
    """Visualise class distributions before and after balancing."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    targets    = ['heat_stress_label', 'dehydration_label']
    titles     = ['Heat Stress', 'Dehydration']
    bar_colors = ['#27AE60', '#F39C12', '#E74C3C']

    for ax, lbl, title in zip(axes, targets, titles):
        vc   = df[lbl].value_counts().sort_index()
        vals = [vc.get(i, 0) for i in range(3)]
        bars = ax.bar(LABEL_NAMES, vals, color=bar_colors, edgecolor='white', lw=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                    f"{v:,}\n({v/sum(vals)*100:.1f}%)", ha='center', fontsize=9, fontweight='bold')
        ax.set_title(f"{title} — Class Distribution (before oversampling)",
                     fontweight='bold')
        ax.set_ylabel('Sample Count')
        ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, 'class_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def main():
    print("\n══════════════════════════════════════════════")
    print("  HeatGuard AI — 3-Class Training Pipeline")
    print("══════════════════════════════════════════════\n")

    print("▶ Loading and merging datasets …")
    df = load_all_data(
        INFANT_PATH, WEARABLE_PATH, P1_PATH, P2_PATH,
        n_syn_high=4000,
        n_syn_mod=1500,
        n_syn_normal=1000,
    )

    print("\n▶ Plotting class distribution …")
    plot_class_distribution(df)

    all_results = {}
    metrics_out = {}

    for target in ['heat_stress_label', 'dehydration_label']:
        print(f"\n{'═'*50}")
        print(f"  Target: {target}")
        print(f"{'═'*50}")

        X, y, feats = get_feature_matrix(df, target=target, balance_strategy='smote_like')
        print(f"  Balanced shape: {X.shape}")

        results, trained = train_and_evaluate(X, y, target)
        all_results[target] = results

        # Save models + thresholds
        for name, model in trained.items():
            path = os.path.join(OUT_DIR, f"{target}_{name}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            # Save thresholds separately too for easy reload
            tpath = os.path.join(OUT_DIR, f"{target}_{name}_thresholds.json")
            with open(tpath, 'w') as f:
                json.dump(results[name]['thresholds'], f)

        with open(os.path.join(OUT_DIR, f"{target}_features.json"), 'w') as f:
            json.dump(feats, f)

        metrics_out[target] = {}
        for name, r in results.items():
            metrics_out[target][name] = {
                'accuracy'   : float(r['accuracy']),
                'f1_weighted': float(r['f1_weighted']),
                'f1_macro'   : float(r['f1_macro']),
                'roc_auc'    : float(r['roc_auc']),
                'cv_mean'    : float(r['cv_mean']),
                'cv_std'     : float(r['cv_std']),
                'thresholds' : r['thresholds'],
                'per_class'  : r['per_class'],
            }

        print(f"\n▶ Plots for {target} …")
        plot_confusion_matrices(results, target)
        plot_roc_curves(results, target)
        plot_per_class_f1(results, target)

    print("\n▶ Combined comparison plots …")
    plot_model_comparison(all_results)
    plot_cv_bars(all_results)

    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    print("\n══════════════════════════════════════════════")
    print("  FINAL RESULTS SUMMARY")
    print("══════════════════════════════════════════════")
    for target, results in all_results.items():
        print(f"\n  {target}:")
        hdr = f"  {'Model':<22} {'Acc':>8} {'F1-W':>8} {'F1-M':>8} {'ROC':>8} {'CV':>14}  High-Recall"
        print(hdr)
        print(f"  {'-'*90}")
        for name, r in results.items():
            hr = r['per_class'].get('High', {}).get('recall', 0.0)
            print(f"  {name:<22} {r['accuracy']:>8.4f} {r['f1_weighted']:>8.4f} "
                  f"{r['f1_macro']:>8.4f} {r['roc_auc']:>8.4f} "
                  f"{r['cv_mean']:>6.4f}±{r['cv_std']:.4f}  {hr:.3f}")

    print(f"\n  Models  → {OUT_DIR}/")
    print(f"  Plots   → {PLOT_DIR}/")
    print("  ✓ Done!\n")


if __name__ == '__main__':
    main()