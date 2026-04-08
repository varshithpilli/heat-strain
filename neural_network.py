"""
neural_network.py
═══════════════════════════════════════════════════════════════
Custom Deep Neural Network — built entirely from scratch with NumPy.
No sklearn MLPClassifier. Pure forward/backward propagation.

Architecture:
  Input → Dense(128) → BN → ReLU → Dropout(0.3)
        → Dense(64)  → BN → ReLU → Dropout(0.2)
        → Dense(32)  → BN → ReLU
        → Dense(n_classes) → Softmax
        

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