"""
Minimal NumPy MLP (ReLU hidden layers, sigmoid output, dropout, Adam, early stopping).

It mirrors the Keras model in `model_mlp.py` and is used ONLY as a fallback when
TensorFlow is not installed, so the whole pipeline still runs end-to-end.
"""
import numpy as np


class NumpyMLP:
    def __init__(self, n_in, hidden=(32, 16), dropout=0.2, lr=1e-3, seed=42):
        self.rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        # He initialisation
        self.W = [self.rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.dropout, self.lr = dropout, lr
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

    def _forward(self, X, train):
        acts, masks = [X], []
        for i in range(len(self.W) - 1):
            a = np.maximum(0, acts[-1] @ self.W[i] + self.b[i])
            if train and self.dropout > 0:
                mask = (self.rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                a = a * mask
                masks.append(mask)
            else:
                masks.append(None)
            acts.append(a)
        out = self._sigmoid(acts[-1] @ self.W[-1] + self.b[-1]).ravel()
        return out, acts, masks

    def predict_proba(self, X):
        return self._forward(X, train=False)[0]

    @staticmethod
    def _bce(y, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _step(self, Xb, yb):
        p, acts, masks = self._forward(Xb, train=True)
        delta = ((p - yb) / len(yb)).reshape(-1, 1)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]
        self.t += 1
        for k, (p_, g) in enumerate(zip(self.W + self.b, gW + gb)):   # Adam update (in place)
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            p_ -= self.lr * mh / (np.sqrt(vh) + 1e-7)

    def fit(self, Xtr, ytr, Xval, yval, epochs=100, batch_size=32, patience=15):
        hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best, best_w, wait = np.inf, None, 0
        for _ in range(epochs):
            idx = self.rng.permutation(len(ytr))
            for s in range(0, len(idx), batch_size):
                b = idx[s:s + batch_size]
                self._step(Xtr[b], ytr[b])
            ptr, pval = self.predict_proba(Xtr), self.predict_proba(Xval)
            hist["loss"].append(self._bce(ytr, ptr)); hist["val_loss"].append(self._bce(yval, pval))
            hist["accuracy"].append(float(np.mean((ptr > .5) == ytr)))
            hist["val_accuracy"].append(float(np.mean((pval > .5) == yval)))
            if hist["val_loss"][-1] < best - 1e-5:
                best, wait = hist["val_loss"][-1], 0
                best_w = ([w.copy() for w in self.W], [b_.copy() for b_ in self.b])
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_w:
            self.W, self.b = best_w
        return hist
