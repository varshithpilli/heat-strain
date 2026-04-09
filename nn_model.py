"""
custom_model.py
═══════════════════════════════════════════════════════════════
Custom Deep Neural Network — built entirely from scratch with NumPy.
No sklearn MLPClassifier. Pure forward/backward propagation.

Architecture:
  Input -> Dense(128) -> BN -> ReLU -> Dropout(0.3)
        -> Dense(64)  -> BN -> ReLU -> Dropout(0.2)
        -> Dense(32)  -> BN -> ReLU
        -> Dense(n_classes) -> Softmax
        

Training:
  • Mini-batch SGD with Adam optimiser
  • L2 weight regularisation
  • Batch normalisation (train/inference modes)
  • Dropout regularisation
  • Early stopping on validation loss
  • Learning rate decay
═══════════════════════════════════════════════════════════════
"""

import numpy as np

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class BatchNorm:
    def __init__(self, dim, eps=1e-8, momentum=0.9):
        self.gamma   = np.ones(dim)
        self.beta    = np.zeros(dim)
        self.eps     = eps
        self.mom     = momentum
        self.run_mu  = np.zeros(dim)
        self.run_var = np.ones(dim)
        
        self.cache   = None

    def forward(self, x, training=True):
        if training:
            mu      = x.mean(axis=0)
            var     = x.var(axis=0)
            x_norm  = (x - mu) / np.sqrt(var + self.eps)
            self.cache = (x, x_norm, mu, var)
            self.run_mu  = self.mom * self.run_mu  + (1 - self.mom) * mu
            self.run_var = self.mom * self.run_var + (1 - self.mom) * var
        else:
            x_norm = (x - self.run_mu) / np.sqrt(self.run_var + self.eps)
        return self.gamma * x_norm + self.beta

    def backward(self, dout):
        x, x_norm, mu, var = self.cache
        N = x.shape[0]
        dgamma = (dout * x_norm).sum(axis=0)
        dbeta  = dout.sum(axis=0)
        dx_norm = dout * self.gamma
        dvar   = (dx_norm * (x - mu) * -0.5 * (var + self.eps) ** -1.5).sum(axis=0)
        dmu    = (dx_norm * -1 / np.sqrt(var + self.eps)).sum(axis=0) \
                 + dvar * (-2 * (x - mu)).sum(axis=0) / N
        dx     = (dx_norm / np.sqrt(var + self.eps)
                  + dvar * 2 * (x - mu) / N + dmu / N)
        return dx, dgamma, dbeta

    def params(self):
        return [self.gamma, self.beta]

    def grads(self, dgamma, dbeta):
        self.gamma_grad = dgamma
        self.beta_grad  = dbeta


class Dense:
    def __init__(self, in_dim, out_dim, l2=1e-4):
        # He initialisation
        scale       = np.sqrt(2.0 / in_dim)
        self.W      = np.random.randn(in_dim, out_dim).astype(np.float64) * scale
        self.b      = np.zeros(out_dim, dtype=np.float64)
        self.l2     = l2
        self.cache  = None
        
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x    = self.cache
        self.dW = x.T @ dout + self.l2 * self.W
        self.db = dout.sum(axis=0)
        return dout @ self.W.T



def adam_step(layer, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    layer.mW = beta1 * layer.mW + (1 - beta1) * layer.dW
    layer.vW = beta2 * layer.vW + (1 - beta2) * layer.dW ** 2
    mW_hat   = layer.mW / (1 - beta1 ** t)
    vW_hat   = layer.vW / (1 - beta2 ** t)
    layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

    layer.mb = beta1 * layer.mb + (1 - beta1) * layer.db
    layer.vb = beta2 * layer.vb + (1 - beta2) * layer.db ** 2
    mb_hat   = layer.mb / (1 - beta1 ** t)
    vb_hat   = layer.vb / (1 - beta2 ** t)
    layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


def adam_step_bn(bn, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    if not hasattr(bn, 'mgamma'):
        bn.mgamma = np.zeros_like(bn.gamma); bn.vgamma = np.zeros_like(bn.gamma)
        bn.mbeta  = np.zeros_like(bn.beta);  bn.vbeta  = np.zeros_like(bn.beta)

    bn.mgamma = beta1 * bn.mgamma + (1 - beta1) * bn.gamma_grad
    bn.vgamma = beta2 * bn.vgamma + (1 - beta2) * bn.gamma_grad ** 2
    mg_hat    = bn.mgamma / (1 - beta1 ** t)
    vg_hat    = bn.vgamma / (1 - beta2 ** t)
    bn.gamma -= lr * mg_hat / (np.sqrt(vg_hat) + eps)

    bn.mbeta  = beta1 * bn.mbeta + (1 - beta1) * bn.beta_grad
    bn.vbeta  = beta2 * bn.vbeta + (1 - beta2) * bn.beta_grad ** 2
    mb_hat    = bn.mbeta  / (1 - beta1 ** t)
    vb_hat    = bn.vbeta  / (1 - beta2 ** t)
    bn.beta  -= lr * mb_hat / (np.sqrt(vb_hat) + eps)



class CustomNeuralNetwork:
    """
    4-layer deep neural network built from scratch.
    Compatible with sklearn Pipeline (fit/predict/predict_proba).
    """

    def __init__(self, hidden_sizes=(128, 64, 32), n_classes=3,
                 lr=1e-3, epochs=200, batch_size=64,
                 l2=1e-4, dropout_rates=(0.3, 0.2, 0.0),
                 patience=20, lr_decay=0.95, random_state=42):
        np.random.seed(random_state)
        self.hidden_sizes   = hidden_sizes
        self.n_classes      = n_classes
        self.lr             = lr
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.l2             = l2
        self.dropout_rates  = dropout_rates
        self.patience       = patience
        self.lr_decay       = lr_decay
        self.random_state   = random_state
        self.layers         = None
        self.bns            = None
        self.scaler_mean    = None
        self.scaler_std     = None
        self.history        = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.classes_       = None

    def _build(self, in_dim):
        self.layers = []
        self.bns    = []
        prev = in_dim
        for h in self.hidden_sizes:
            self.layers.append(Dense(prev, h, l2=self.l2))
            self.bns.append(BatchNorm(h))
            prev = h
        
        self.layers.append(Dense(prev, self.n_classes, l2=self.l2))

    def _normalise(self, X, fit=False):
        if fit:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std  = X.std(axis=0).clip(min=1e-8)
        return (X - self.scaler_mean) / self.scaler_std

    def _forward(self, X, training=True, dropout_masks=None):
        a = X.astype(np.float64)
        if dropout_masks is None:
            dropout_masks = []
        caches = []
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.bns)):
            z    = layer.forward(a)
            z_bn = bn.forward(z, training=training)
            a    = relu(z_bn)
            caches.append((z, z_bn, a))

            dr = self.dropout_rates[i] if i < len(self.dropout_rates) else 0.0
            if training and dr > 0:
                if len(dropout_masks) <= i:
                    mask = (np.random.rand(*a.shape) > dr).astype(np.float64) / (1 - dr)
                    dropout_masks.append(mask)
                a = a * dropout_masks[i]
            elif not training and dr > 0:
                pass 
        logits = self.layers[-1].forward(a)
        probs  = softmax(logits)
        return probs, caches, dropout_masks

    def _loss(self, probs, y):
        N     = y.shape[0]
        eps   = 1e-12
        loss  = -np.log(probs[np.arange(N), y] + eps).mean()
        
        l2_pen = sum(0.5 * self.l2 * (layer.W ** 2).sum()
                     for layer in self.layers)
        return loss + l2_pen / N

    def _backward(self, probs, y, caches, dropout_masks, t):
        N    = y.shape[0]
        lr   = self.lr * (self.lr_decay ** (t // 10))
        lr   = max(lr, 1e-5)

        d = probs.copy()
        d[np.arange(N), y] -= 1
        d /= N

        dx = self.layers[-1].backward(d)
        adam_step(self.layers[-1], lr, t)

        for i in range(len(self.bns) - 1, -1, -1):
            z, z_bn, a = caches[i]

            dr = self.dropout_rates[i] if i < len(self.dropout_rates) else 0.0
            if dr > 0 and i < len(dropout_masks):
                dx = dx * dropout_masks[i]

            dx = dx * relu_grad(z_bn)

            dx, dgamma, dbeta = self.bns[i].backward(dx)
            self.bns[i].grads(dgamma, dbeta)
            adam_step_bn(self.bns[i], lr, t)

           
            dx = self.layers[i].backward(dx)
            adam_step(self.layers[i], lr, t)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        X = self._normalise(X, fit=True)
        self._build(X.shape[1])

        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, random_state=self.random_state, stratify=y)

        best_val_loss = np.inf
        no_improve    = 0
        best_weights  = None
        t             = 0

        n = len(X_tr)
        for epoch in range(1, self.epochs + 1):
            
            idx = np.random.permutation(n)
            X_tr, y_tr = X_tr[idx], y_tr[idx]

            
            for start in range(0, n, self.batch_size):
                t += 1
                Xb = X_tr[start:start + self.batch_size]
                yb = y_tr[start:start + self.batch_size]
                masks = []
                probs, caches, masks = self._forward(Xb, training=True,
                                                     dropout_masks=masks)
                self._backward(probs, yb, caches, masks, t)

            
            p_val, _, _ = self._forward(X_val, training=False)
            val_loss    = self._loss(p_val, y_val)
            val_acc     = (p_val.argmax(axis=1) == y_val).mean()
            p_tr, _, _  = self._forward(X_tr, training=False)
            tr_loss     = self._loss(p_tr, y_tr)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                no_improve    = 0
                best_weights  = self._copy_weights()
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"    Early stop @ epoch {epoch}  val_acc={val_acc:.4f}")
                    break

            if epoch % 20 == 0:
                print(f"    Epoch {epoch:4d}  tr_loss={tr_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if best_weights:
            self._load_weights(best_weights)
        return self

    def predict(self, X):
        X = self._normalise(X)
        probs, _, _ = self._forward(X, training=False)
        return probs.argmax(axis=1)

    def predict_proba(self, X):
        X = self._normalise(X)
        probs, _, _ = self._forward(X, training=False)
        return probs

    def _copy_weights(self):
        snap = {'layers': [], 'bns': []}
        for l in self.layers:
            snap['layers'].append((l.W.copy(), l.b.copy()))
        for bn in self.bns:
            snap['bns'].append((bn.gamma.copy(), bn.beta.copy(),
                                bn.run_mu.copy(), bn.run_var.copy()))
        return snap

    def _load_weights(self, snap):
        for l, (W, b) in zip(self.layers, snap['layers']):
            l.W, l.b = W, b
        for bn, (g, bt, rm, rv) in zip(self.bns, snap['bns']):
            bn.gamma, bn.beta, bn.run_mu, bn.run_var = g, bt, rm, rv

    
    def get_params(self, deep=True):
        return dict(hidden_sizes=self.hidden_sizes, n_classes=self.n_classes,
                    lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                    l2=self.l2, dropout_rates=self.dropout_rates,
                    patience=self.patience, lr_decay=self.lr_decay,
                    random_state=self.random_state)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ═══════════════════════════════════════════════════════════════
# Training, Evaluation & Export Pipeline
# ═══════════════════════════════════════════════════════════════

import os, sys, json, pickle, warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc,
    recall_score, precision_score
)

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import get_feature_matrix, LABEL_NAMES

OUT_DIR  = 'saved_models'
PLOT_DIR = 'plots'
MODEL_KEY = 'custom_nn'


def calibrate_thresholds(model, X_val, y_val, n_classes=3):
    """
    Find per-class probability thresholds that maximise
    macro F1 on the validation set.
    """
    proba = model.predict_proba(X_val)
    best_f1, best_thresh = 0.0, np.full(n_classes, 1.0 / n_classes)

    for t0 in np.arange(0.20, 0.65, 0.05):
        for t1 in np.arange(0.20, 0.65, 0.05):
            t2 = 1.0 - t0 - t1
            if t2 < 0.05 or t2 > 0.80:
                continue
            thresh = np.array([t0, t1, t2])
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


def _cmap_purple():
    return LinearSegmentedColormap.from_list('p', ['#EEEDFE', '#534AB7'])


def plot_nn_confusion(y_test, y_pred, labels, target):
    cm = confusion_matrix(y_test, y_pred)
    cn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(cn, cmap=_cmap_purple(), vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    ax.set_title(f"Custom Neural Network\nAcc={acc:.3f}  F1={f1m:.3f}",
                 fontweight='bold', fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tc = 'white' if cn[i, j] > 0.5 else '#2C2C2A'
            ax.text(j, i, f"{cm[i,j]}\n({cn[i,j]:.2f})",
                    ha='center', va='center', fontsize=8, color=tc)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'nn_confusion_{target}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def plot_nn_roc(y_test, y_prob, classes, target):
    fig, ax = plt.subplots(figsize=(8, 6))
    color = '#7F77DD'
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"Custom NN AUC={auc(fpr, tpr):.3f}")
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
                label=f"Custom NN AUC={auc(all_fpr, mean_tpr):.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    title_name = target.replace('_label', '').replace('_', ' ').title()
    ax.set_title(f"ROC Curve — Custom NN — {title_name}",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'nn_roc_{target}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def plot_nn_per_class_f1(per_class, target):
    classes_list = list(per_class.keys())
    class_colors = {'Normal': '#27AE60', 'Moderate': '#F39C12', 'High': '#E74C3C'}
    fig, ax = plt.subplots(figsize=(5, 4.5))
    vals = [per_class[c]['f1'] for c in classes_list]
    cols = [class_colors.get(c, '#888') for c in classes_list]
    bars = ax.bar(classes_list, vals, color=cols, edgecolor='white', lw=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.15)
    title_name = target.replace('_label', '').replace('_', ' ').title()
    ax.set_title(f"Custom Neural Network — Per-Class F1\n{title_name}",
                 fontweight='bold', fontsize=10)
    ax.axhline(0.80, color='grey', linestyle='--', lw=1, alpha=0.5)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'nn_per_class_f1_{target}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def main():
    print("\n══════════════════════════════════════════════")
    print("  HeatGuard AI — Custom Neural Network Pipeline")
    print("══════════════════════════════════════════════\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv("combined_dataset.csv")

    metrics_out = {}

    for target in ['heat_stress_label', 'dehydration_label']:
        print(f"\n{'═'*50}")
        print(f"  Target: {target}")
        print(f"{'═'*50}")

        X, y, feats = get_feature_matrix(df, target=target, balance_strategy='smote_like')
        X = X.astype(np.float32)
        print(f"  Balanced shape: {X.shape}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)

        classes = sorted(np.unique(y))
        labels  = [LABEL_NAMES[i] for i in classes]

        print(f"\n  Test set class distribution:")
        for cls in classes:
            cnt = (y_te == cls).sum()
            print(f"    {LABEL_NAMES[cls]}: {cnt} ({cnt/len(y_te)*100:.1f}%)")

        print(f"\n  ── [custom_nn] ──")
        model = CustomNeuralNetwork(
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
        )
        model.fit(X_tr2, y_tr2)

        thresholds = calibrate_thresholds(model, X_val, y_val)
        print(f"    Thresholds: {thresholds.round(3)}")

        yp, yprob = predict_with_thresholds(model, X_te, thresholds)

        acc    = accuracy_score(y_te, yp)
        f1     = f1_score(y_te, yp, average='weighted', zero_division=0)
        f1m    = f1_score(y_te, yp, average='macro',    zero_division=0)
        p_cls  = precision_score(y_te, yp, average=None, labels=classes, zero_division=0)
        r_cls  = recall_score(   y_te, yp, average=None, labels=classes, zero_division=0)
        f1_cls = f1_score(       y_te, yp, average=None, labels=classes, zero_division=0)

        if len(classes) > 2:
            yb  = label_binarize(y_te, classes=classes)
            roc = roc_auc_score(yb, yprob, multi_class='ovr', average='weighted')
        else:
            roc = roc_auc_score(y_te, yprob[:, 1])

        print(f"    Accuracy={acc:.4f}  F1-weighted={f1:.4f}  F1-macro={f1m:.4f}  ROC={roc:.4f}")
        for i, cls in enumerate(classes):
            print(f"    {LABEL_NAMES[cls]:8s}: P={p_cls[i]:.3f}  R={r_cls[i]:.3f}  F1={f1_cls[i]:.3f}")

        # ── Cross-validation ──
        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_sc = []
        for fold, (tr_i, val_i) in enumerate(skf.split(X, y)):
            m_cv = CustomNeuralNetwork(
                hidden_sizes=(128, 64, 32), n_classes=3, lr=1e-3,
                epochs=200, batch_size=64, l2=1e-4,
                dropout_rates=(0.3, 0.2, 0.0), patience=20,
                lr_decay=0.97, random_state=42,
            )
            m_cv.fit(X[tr_i], y[tr_i])
            th_cv = calibrate_thresholds(m_cv, X[val_i], y[val_i])
            yp_cv, _ = predict_with_thresholds(m_cv, X[val_i], th_cv)
            cv_sc.append(accuracy_score(y[val_i], yp_cv))
        cv_mean, cv_std = np.mean(cv_sc), np.std(cv_sc)
        print(f"    CV: {cv_mean:.4f} ± {cv_std:.4f}")

        per_class = {LABEL_NAMES[cls]: {
            'precision': float(p_cls[i]),
            'recall'   : float(r_cls[i]),
            'f1'       : float(f1_cls[i]),
        } for i, cls in enumerate(classes)}

        # ── Save model (.pkl) ──
        model._heatguard_thresholds = thresholds
        path = os.path.join(OUT_DIR, f"{target}_{MODEL_KEY}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved model → {path}")

        # ── Save thresholds (.json) ──
        tpath = os.path.join(OUT_DIR, f"{target}_{MODEL_KEY}_thresholds.json")
        with open(tpath, 'w') as f:
            json.dump(thresholds.tolist(), f)

        # ── Save features (.json) ──
        with open(os.path.join(OUT_DIR, f"{target}_features.json"), 'w') as f:
            json.dump(feats, f)

        # ── Collect metrics ──
        metrics_out[target] = {
            MODEL_KEY: {
                'accuracy'   : float(acc),
                'f1_weighted': float(f1),
                'f1_macro'   : float(f1m),
                'roc_auc'    : float(roc),
                'cv_mean'    : float(cv_mean),
                'cv_std'     : float(cv_std),
                'thresholds' : thresholds.tolist(),
                'per_class'  : per_class,
            }
        }

        # ── Plots ──
        print(f"\n▶ Plots for {target} …")
        plot_nn_confusion(y_te, yp, labels, target)
        plot_nn_roc(y_te, yprob, classes, target)
        plot_nn_per_class_f1(per_class, target)

    # ── Save combined metrics (.json) ──
    with open(os.path.join(OUT_DIR, 'nn_metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # ── Summary ──
    print("\n══════════════════════════════════════════════")
    print("  FINAL RESULTS SUMMARY — Custom Neural Network")
    print("══════════════════════════════════════════════")
    for target, tdata in metrics_out.items():
        print(f"\n  {target}:")
        r  = tdata[MODEL_KEY]
        hr = r['per_class'].get('High', {}).get('recall', 0.0)
        hdr = f"  {'Model':<22} {'Acc':>8} {'F1-W':>8} {'F1-M':>8} {'ROC':>8} {'CV':>14}  High-Recall"
        print(hdr)
        print(f"  {'-'*90}")
        print(f"  {MODEL_KEY:<22} {r['accuracy']:>8.4f} {r['f1_weighted']:>8.4f} "
              f"{r['f1_macro']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['cv_mean']:>6.4f}±{r['cv_std']:.4f}  {hr:.3f}")

    print(f"\n  Models  → {OUT_DIR}/")
    print(f"  Plots   → {PLOT_DIR}/")
    print("  ✓ Done!\n")


if __name__ == '__main__':
    main()