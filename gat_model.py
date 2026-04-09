"""
gat_model.py
═══════════════════════════════════════════════════════════════
Graph Attention Network (GAT) for physiological stress classification.
Architecture:
  GATConv(3→64, heads=8) → GATConv(512→32, heads=4) → skip-add
  → GlobalMeanPool → Dense(32) → BN → ReLU
  → heat_head(3)  +  dehyd_head(2)   [multi-task]

Training:
  • Joint multi-task loss (heat + dehydration heads)
  • Class-weighted cross-entropy for heat stress
  • Adam + StepLR scheduler (step=10, γ=0.7)
  • 5-fold StratifiedKFold cross-validation
  • Per-class threshold calibration
  • Correlation-based graph edges (k=4)
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score
)

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import get_feature_matrix, LABEL_NAMES

OUT_DIR   = "saved_models"
PLOT_DIR  = "plots"
MODEL_KEY = "gat"

# Class weights for heat stress imbalance (Normal / Moderate / High)
HEAT_WEIGHTS = torch.tensor([1.0, 1.3, 1.6])


# ═══════════════════════════════════════════════════════════════
# Model  — dual-head multi-task GAT
# ═══════════════════════════════════════════════════════════════

class PhysioGAT(torch.nn.Module):
    """
    Single backbone, two classification heads:
      heat_head  → 3 classes (Normal / Moderate / High)
      dehyd_head → 2 classes (Normal / At Risk)
    Both heads are trained jointly every batch.
    """
    def __init__(self):
        super().__init__()
        self.gat1      = GATConv(3, 64, heads=8, dropout=0.3)
        self.gat2      = GATConv(64 * 8, 32, heads=4, dropout=0.3)
        self.fc        = Linear(32 * 4, 32)
        self.bn        = BatchNorm1d(32)
        self.heat_head = Linear(32, 3)
        self.dehyd_head= Linear(32, 2)

    def forward(self, x, edge_index, batch):
        x1 = F.relu(self.gat1(x, edge_index))
        x2 = F.relu(self.gat2(x1, edge_index))
        x  = x1[:, :x2.shape[1]] + x2          # residual / skip
        x  = global_mean_pool(x, batch)
        x  = F.relu(self.bn(self.fc(x)))
        return self.heat_head(x), self.dehyd_head(x)


# ═══════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════

def build_edge_index(X, k=4):
    corr  = np.corrcoef(X.T)
    edges = []
    for i in range(corr.shape[0]):
        strongest = np.argsort(-np.abs(corr[i]))[1:k + 1]
        for j in strongest:
            edges.append((i, j))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_graphs(X, y_heat, y_dehyd, edge_index):
    """Each sample becomes one graph; nodes = features."""
    graphs = []
    for i in range(len(X)):
        node_features = torch.tensor(
            np.stack([X[i], X[i] ** 2, np.abs(X[i])], axis=1),
            dtype=torch.float
        )
        graphs.append(Data(
            x         = node_features,
            edge_index= edge_index,
            y_heat    = torch.tensor(y_heat[i],  dtype=torch.long),
            y_dehyd   = torch.tensor(y_dehyd[i], dtype=torch.long),
        ))
    return graphs


# ═══════════════════════════════════════════════════════════════
# Threshold calibration
# ═══════════════════════════════════════════════════════════════

def calibrate_thresholds(model, loader, device, target, n_classes):
    """Maximise macro-F1 on a held-out validation loader."""
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            h_logit, d_logit = model(batch.x, batch.edge_index, batch.batch)
            logits = h_logit if target == "heat_stress_label" else d_logit
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            labels = (batch.y_heat if target == "heat_stress_label"
                      else batch.y_dehyd).cpu().numpy()
            y_true.extend(labels)
            y_prob.extend(probs)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    best_f1, best_thresh = 0.0, np.full(n_classes, 1.0 / n_classes)

    if n_classes == 2:
        for t0 in np.arange(0.20, 0.80, 0.05):
            thresh = np.array([t0, 1.0 - t0])
            adj    = y_prob / (thresh + 1e-9)
            f1     = f1_score(y_true, adj.argmax(axis=1), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh.copy()
    else:
        for t0 in np.arange(0.20, 0.65, 0.05):
            for t1 in np.arange(0.20, 0.65, 0.05):
                t2 = 1.0 - t0 - t1
                if t2 < 0.05 or t2 > 0.80:
                    continue
                thresh = np.array([t0, t1, t2])
                thresh = thresh / thresh.sum()
                adj    = y_prob / (thresh + 1e-9)
                f1     = f1_score(y_true, adj.argmax(axis=1), average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thresh = f1, thresh.copy()

    return best_thresh


def predict_with_thresholds(model, loader, device, target, thresholds):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            h_logit, d_logit = model(batch.x, batch.edge_index, batch.batch)
            logits = h_logit if target == "heat_stress_label" else d_logit
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            labels = (batch.y_heat if target == "heat_stress_label"
                      else batch.y_dehyd).cpu().numpy()
            adj    = probs / (thresholds + 1e-9)
            y_true.extend(labels)
            y_pred.extend(adj.argmax(axis=1))
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ═══════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════

def _cmap_purple():
    return LinearSegmentedColormap.from_list("p", ["#EEEDFE", "#534AB7"])


def plot_gat_confusion(y_test, y_pred, labels, target):
    cm = confusion_matrix(y_test, y_pred)
    cn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(cn, cmap=_cmap_purple(), vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    ax.set_title(f"Graph Attention Network\nAcc={acc:.3f}  F1={f1m:.3f}",
                 fontweight="bold", fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tc = "white" if cn[i, j] > 0.5 else "#2C2C2A"
            ax.text(j, i, f"{cm[i,j]}\n({cn[i,j]:.2f})",
                    ha="center", va="center", fontsize=8, color=tc)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"gat_confusion_{target}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


def plot_gat_roc(y_test, y_prob, classes, target):
    fig, ax = plt.subplots(figsize=(8, 6))
    color   = "#7F77DD"
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"GAT AUC={auc(fpr, tpr):.3f}")
    else:
        yb       = label_binarize(y_test, classes=classes)
        all_fpr  = np.unique(np.concatenate(
            [roc_curve(yb[:, i], y_prob[:, i])[0] for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            fpr_i, tpr_i, _ = roc_curve(yb[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= len(classes)
        ax.plot(all_fpr, mean_tpr, color=color, lw=2.5,
                label=f"GAT AUC={auc(all_fpr, mean_tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    title_name = target.replace("_label", "").replace("_", " ").title()
    ax.set_title(f"ROC Curve — GAT — {title_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"gat_roc_{target}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


def plot_gat_per_class_f1(per_class, target):
    classes_list = list(per_class.keys())
    class_colors = {"Normal": "#27AE60", "Moderate": "#F39C12",
                    "High": "#E74C3C", "At Risk": "#E67E22"}
    fig, ax = plt.subplots(figsize=(5, 4.5))
    vals = [per_class[c]["f1"] for c in classes_list]
    cols = [class_colors.get(c, "#888") for c in classes_list]
    bars = ax.bar(classes_list, vals, color=cols, edgecolor="white", lw=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)
    title_name = target.replace("_label", "").replace("_", " ").title()
    ax.set_title(f"GAT — Per-Class F1\n{title_name}", fontweight="bold", fontsize=10)
    ax.axhline(0.80, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"gat_per_class_f1_{target}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════
# Core training — one shared model, both targets at once
# ═══════════════════════════════════════════════════════════════

def train_model(df, device):
    """
    Train a single PhysioGAT on both targets jointly.
    Returns the trained model, graph loaders, raw arrays, and feature list.
    """
    print("\n  Loading & standardising data...")

    from data_preprocessing import BASE_FEATURES
    feats = [f for f in BASE_FEATURES if f in df.columns]
    
    df_clean = df.dropna(subset=feats + ["heat_stress_label", "dehydration_label"])
    X = df_clean[feats].values.astype(np.float32)
    y_heat = np.clip(df_clean["heat_stress_label"].values.astype(int), 0, 2)
    y_dehyd = np.clip(df_clean["dehydration_label"].values.astype(int), 0, 1)

    print(f"  Aligned dataset shape: {X.shape}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n = len(X)
    # ── Splits ──
    idx = np.arange(n)
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_heat)
    tr_idx, val_idx = train_test_split(tr_idx, test_size=0.15, random_state=42, stratify=y_heat[tr_idx])

    X_tr, y_h_tr, y_d_tr   = X[tr_idx],  y_heat[tr_idx],  y_dehyd[tr_idx]
    X_val, y_h_val, y_d_val = X[val_idx], y_heat[val_idx], y_dehyd[val_idx]
    X_te, y_h_te, y_d_te   = X[te_idx],  y_heat[te_idx],  y_dehyd[te_idx]

    # ── Graphs ──
    edge_index   = build_edge_index(X_tr)
    train_graphs = build_graphs(X_tr,  y_h_tr,  y_d_tr,  edge_index)
    val_graphs   = build_graphs(X_val, y_h_val, y_d_val, edge_index)
    test_graphs  = build_graphs(X_te,  y_h_te,  y_d_te,  edge_index)

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_graphs,   batch_size=64, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_graphs,  batch_size=64, shuffle=False, num_workers=0)

    # ── Model ──
    model     = PhysioGAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    hw        = HEAT_WEIGHTS.to(device)
    epochs    = 60

    print(f"\n  ── [gat — multi-task] ──")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            h_out, d_out = model(batch.x, batch.edge_index, batch.batch)
            loss = (F.cross_entropy(h_out, batch.y_heat,  weight=hw) +
                    F.cross_entropy(d_out, batch.y_dehyd))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  Loss={total_loss:.4f}")

    return model, train_loader, val_loader, test_loader, edge_index, \
           X, y_heat, y_dehyd, feats


# ═══════════════════════════════════════════════════════════════
# Per-target evaluation + CV
# ═══════════════════════════════════════════════════════════════

def evaluate_target(model, val_loader, test_loader,
                    X_full, y_heat_full, y_dehyd_full,
                    target, device):

    n_classes = 3 if target == "heat_stress_label" else 2
    y_full    = y_heat_full if target == "heat_stress_label" else y_dehyd_full
    classes   = sorted(np.unique(y_full))
    labels    = [LABEL_NAMES[i] for i in classes]

    print(f"\n{'═' * 50}")
    print(f"  Evaluating: {target}")
    print(f"{'═' * 50}")

    # ── Threshold calibration on val ──
    thresholds = calibrate_thresholds(model, val_loader, device, target, n_classes)
    print(f"    Thresholds: {thresholds.round(3)}")

    # ── Test evaluation ──
    y_true, y_pred, y_prob = predict_with_thresholds(
        model, test_loader, device, target, thresholds)
    y_pred_argmax = y_prob.argmax(axis=1)

    print("Accuracy (with thresholds):", accuracy_score(y_true, y_pred))
    print("Accuracy (plain argmax):   ", accuracy_score(y_true, y_pred_argmax))

    acc   = accuracy_score(y_true, y_pred)
    f1w   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    p_cls = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    r_cls = recall_score(   y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1_cls= f1_score(       y_true, y_pred, average=None, labels=classes, zero_division=0)

    if n_classes > 2:
        yb  = label_binarize(y_true, classes=classes)
        roc = roc_auc_score(yb, y_prob, multi_class="ovr", average="weighted")
    else:
        roc = roc_auc_score(y_true, y_prob[:, 1])

    print(f"    Accuracy={acc:.4f}  F1-weighted={f1w:.4f}  F1-macro={f1m:.4f}  ROC={roc:.4f}")
    for i, cls in enumerate(classes):
        print(f"    {LABEL_NAMES[cls]:8s}: P={p_cls[i]:.3f}  R={r_cls[i]:.3f}  F1={f1_cls[i]:.3f}")

    # ── Cross-validation ──
    print("  Running 5-fold cross-validation...")
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    hw        = HEAT_WEIGHTS.to(device)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_full, y_full)):
        print(f"    Fold {fold + 1}/5...", end=" ", flush=True)

        y_h_tr  = y_heat_full[tr_i];  y_d_tr  = y_dehyd_full[tr_i]
        y_h_val = y_heat_full[val_i]; y_d_val = y_dehyd_full[val_i]

        edge_cv  = build_edge_index(X_full[tr_i])
        gtr_cv   = build_graphs(X_full[tr_i],  y_h_tr,  y_d_tr,  edge_cv)
        gval_cv  = build_graphs(X_full[val_i], y_h_val, y_d_val, edge_cv)

        tl_cv = DataLoader(gtr_cv,  batch_size=64, shuffle=True,  num_workers=0)
        vl_cv = DataLoader(gval_cv, batch_size=64, shuffle=False, num_workers=0)

        m_cv  = PhysioGAT().to(device)
        opt_cv = torch.optim.Adam(m_cv.parameters(), lr=0.001)
        sch_cv = torch.optim.lr_scheduler.StepLR(opt_cv, step_size=10, gamma=0.7)

        for _ in range(20):
            m_cv.train()
            for b in tl_cv:
                b = b.to(device)
                opt_cv.zero_grad()
                h_out, d_out = m_cv(b.x, b.edge_index, b.batch)
                loss = (F.cross_entropy(h_out, b.y_heat,  weight=hw) +
                        F.cross_entropy(d_out, b.y_dehyd))
                loss.backward()
                opt_cv.step()
            sch_cv.step()

        th_cv = calibrate_thresholds(m_cv, vl_cv, device, target, n_classes)
        yt_cv, yp_cv, _ = predict_with_thresholds(m_cv, vl_cv, device, target, th_cv)
        fold_acc = accuracy_score(yt_cv, yp_cv)
        cv_scores.append(fold_acc)
        print(f"acc={fold_acc:.4f}")

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    print(f"    CV complete. Mean={cv_mean:.4f} ± {cv_std:.4f}")

    per_class = {}
    for i, cls in enumerate(classes):
        name = LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(cls)
        per_class[name] = {
            "precision": float(p_cls[i]),
            "recall":    float(r_cls[i]),
            "f1":        float(f1_cls[i]),
        }

    return {
        "accuracy":    float(acc),
        "f1_weighted": float(f1w),
        "f1_macro":    float(f1m),
        "roc_auc":     float(roc),
        "cv_mean":     cv_mean,
        "cv_std":      cv_std,
        "thresholds":  thresholds.tolist(),
        "per_class":   per_class,
    }, y_true, y_pred, y_prob, classes, labels, thresholds


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n══════════════════════════════════════════════")
    print("  HeatGuard AI — Graph Attention Network Pipeline")
    print("══════════════════════════════════════════════\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    df     = pd.read_csv("combined_dataset.csv")
    device = torch.device("cpu")

    # ── Train once, both heads ──
    (model, train_loader, val_loader, test_loader,
     edge_index, X_full, y_heat_full, y_dehyd_full, feats) = train_model(df, device)

    metrics_out = {}

    for target in ["heat_stress_label", "dehydration_label"]:
        result, y_true, y_pred, y_prob, classes, labels, thresholds = evaluate_target(
            model, val_loader, test_loader,
            X_full, y_heat_full, y_dehyd_full,
            target, device
        )
        metrics_out[target] = {MODEL_KEY: result}

        # ── Save model (once, tagged with target for compatibility) ──
        model._heatguard_thresholds = thresholds
        model_path = os.path.join(OUT_DIR, f"{target}_{MODEL_KEY}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved model → {model_path}")

        tpath = os.path.join(OUT_DIR, f"{target}_{MODEL_KEY}_thresholds.json")
        with open(tpath, "w") as f:
            json.dump(thresholds.tolist(), f)

        with open(os.path.join(OUT_DIR, f"{target}_gat_features.json"), "w") as f:
            json.dump(feats, f)

        # ── Plots ──
        print(f"\n▶ Plots for {target} …")
        plot_gat_confusion(y_true, y_pred, labels, target)
        plot_gat_roc(y_true, y_prob, classes, target)
        plot_gat_per_class_f1(result["per_class"], target)

    # ── Save metrics ──
    with open(os.path.join(OUT_DIR, "gat_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Summary ──
    print("\n══════════════════════════════════════════════")
    print("  FINAL RESULTS SUMMARY — GAT")
    print("══════════════════════════════════════════════")
    for target, tdata in metrics_out.items():
        r  = tdata[MODEL_KEY]
        hr = r["per_class"].get("High", {}).get("recall", 0.0)
        print(f"\n  {target}:")
        print(f"  {'Model':<22} {'Acc':>8} {'F1-W':>8} {'F1-M':>8} {'ROC':>8} {'CV':>14}  High-Recall")
        print(f"  {'-' * 90}")
        print(f"  {MODEL_KEY:<22} {r['accuracy']:>8.4f} {r['f1_weighted']:>8.4f} "
              f"{r['f1_macro']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['cv_mean']:>6.4f}±{r['cv_std']:.4f}  {hr:.3f}")

    print(f"\n  Models → {OUT_DIR}/")
    print(f"  Plots  → {PLOT_DIR}/")
    print("  ✓ Done!\n")


if __name__ == "__main__":
    main()