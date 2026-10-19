"""
Steps 4-5: MLP model (Keras/TensorFlow), basic hyper-parameter tuning,
feature importance and evaluation visualisations.

Run:  python src/model_mlp.py          (after src/eda_visualization.py)

If TensorFlow is not installed, an equivalent NumPy MLP (numpy_mlp.py) is used and
the backend name is stored in results/metrics.json.
"""
import itertools
import json
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)

from config import CLASS_NAMES, FIG_DIR, PROC_DIR, RESULTS_DIR, SEED, TARGET
from numpy_mlp import NumpyMLP

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
np.random.seed(SEED); random.seed(SEED)

try:                                     # ---- backend selection
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    BACKEND = f"Keras/TensorFlow {tf.__version__}"
except ImportError:
    BACKEND = "NumPy MLP (TensorFlow not installed)"

EPOCHS, BATCH, PATIENCE = 100, 32, 15


def load_data():
    tr, va, te = (pd.read_csv(PROC_DIR / f"{n}.csv") for n in ("train", "val", "test"))
    feats = [c for c in tr.columns if c != TARGET]
    return (tr[feats].values, tr[TARGET].values.astype(float), va[feats].values, va[TARGET].values.astype(float),
            te[feats].values, te[TARGET].values.astype(float), feats)


class Model:
    """Thin wrapper giving Keras and NumPy models the same interface."""

    def __init__(self, n_in, hidden, dropout, lr):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        if BACKEND.startswith("Keras"):
            keras.backend.clear_session()
            layers = [keras.layers.Input((n_in,))]
            for h in hidden:                       # Dense -> ReLU -> Dropout blocks
                layers += [keras.layers.Dense(h, activation="relu"), keras.layers.Dropout(dropout)]
            layers.append(keras.layers.Dense(1, activation="sigmoid"))
            self.m = keras.Sequential(layers)
            self.m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
        else:
            self.m = NumpyMLP(n_in, hidden, dropout, lr, SEED)

    def fit(self, Xtr, ytr, Xva, yva):
        if BACKEND.startswith("Keras"):
            es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
            h = self.m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=0)
            return {k: [float(x) for x in v] for k, v in h.history.items()}
        return self.m.fit(Xtr, ytr, Xva, yva, EPOCHS, BATCH, PATIENCE)

    def predict_proba(self, X):
        if BACKEND.startswith("Keras"):
            return self.m.predict(X, verbose=0).ravel()
        return self.m.predict_proba(X)

    def summary_text(self):
        if BACKEND.startswith("Keras"):
            lines = []
            self.m.summary(print_fn=lines.append)
            return "\n".join(lines)
        return f"NumPy MLP: {self.hidden}, dropout={self.dropout}, lr={self.lr}"


def tune(Xtr, ytr, Xva, yva):
    """Basic grid search on the VALIDATION set (test set is never touched)."""
    grid = {"hidden": [(16, 8), (32, 16), (64, 32)], "dropout": [0.2, 0.3], "lr": [1e-3, 1e-2]}
    rows = []
    for hidden, dr, lr in itertools.product(grid["hidden"], grid["dropout"], grid["lr"]):
        mdl = Model(Xtr.shape[1], hidden, dr, lr)
        hist = mdl.fit(Xtr, ytr, Xva, yva)
        rows.append({"hidden_units": str(hidden), "dropout": dr, "learning_rate": lr,
                     "best_val_loss": round(min(hist["val_loss"]), 4),
                     "best_val_acc": round(max(hist["val_accuracy"] if "val_accuracy" in hist else hist["val_acc"]), 4),
                     "epochs_run": len(hist["loss"])})
        print(rows[-1])
    res = pd.DataFrame(rows).sort_values("best_val_loss").reset_index(drop=True)
    res.to_csv(RESULTS_DIR / "hyperparameter_tuning.csv", index=False)
    best = res.iloc[0]
    return eval(best["hidden_units"]), float(best["dropout"]), float(best["learning_rate"]), res


def permutation_importance(model, X, y, feats, n_rep=20):
    """Drop in accuracy when a feature is shuffled (model-agnostic feature importance)."""
    rng = np.random.default_rng(SEED)
    base = accuracy_score(y, model.predict_proba(X) > .5)
    imp = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_rep):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - accuracy_score(y, model.predict_proba(Xp) > .5))
        imp.append(np.mean(drops))
    return pd.Series(imp, index=feats).sort_values(ascending=False)


def plots(hist, y_te, p_te, imp, rf_imp):
    ep = np.arange(1, len(hist["loss"]) + 1)
    acc_k, vacc_k = ("accuracy", "val_accuracy") if "val_accuracy" in hist else ("acc", "val_acc")

    fig, ax = plt.subplots(figsize=(7, 4.5))                                 # R1 loss
    ax.plot(ep, hist["loss"], label="Training loss"); ax.plot(ep, hist["val_loss"], label="Validation loss")
    ax.axvline(int(np.argmin(hist["val_loss"])) + 1, color="grey", ls="--", label="Best epoch")
    ax.set(title="Loss vs Epoch", xlabel="Epoch", ylabel="Binary cross-entropy"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "r1_loss_vs_epoch.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))                                 # R2 accuracy
    ax.plot(ep, hist[acc_k], label="Training accuracy"); ax.plot(ep, hist[vacc_k], label="Validation accuracy")
    ax.set(title="Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy", ylim=(.85, 1.005)); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "r2_accuracy_vs_epoch.png", dpi=150); plt.close(fig)

    cm = confusion_matrix(y_te, p_te > .5)                                   # R3 confusion matrix
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False,
                annot_kws={"size": 16}, ax=ax)
    ax.set(title="Confusion matrix (test set)", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(FIG_DIR / "r3_confusion_matrix.png", dpi=150); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_te, p_te); roc_auc = auc(fpr, tpr)             # R4 ROC
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot(fpr, tpr, lw=2, label=f"MLP (AUC = {roc_auc:.3f})"); ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "r4_roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))                          # R5 feature importance
    t = imp.head(10)[::-1]; ax[0].barh(t.index, t.values, color="#e76f51"); ax[0].set_title("MLP permutation importance (top 10)")
    ax[0].set_xlabel("Mean drop in validation accuracy")
    t = rf_imp.head(10)[::-1]; ax[1].barh(t.index, t.values, color="#2a9d8f"); ax[1].set_title("Random Forest importance (top 10)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "r5_feature_importance.png", dpi=150); plt.close(fig)
    return float(roc_auc)


def main():
    Xtr, ytr, Xva, yva, Xte, yte, feats = load_data()
    print("Backend:", BACKEND)

    hidden, dropout, lr, tuning = tune(Xtr, ytr, Xva, yva)                   # hyper-parameter tuning
    model = Model(Xtr.shape[1], hidden, dropout, lr)                         # final model
    hist = model.fit(Xtr, ytr, Xva, yva)

    p_te = model.predict_proba(Xte); pred = (p_te > .5).astype(int)
    imp = permutation_importance(model, Xva, yva, feats)
    rf = RandomForestClassifier(300, random_state=SEED).fit(Xtr, ytr)
    rf_imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    roc_auc = plots(hist, yte, p_te, imp, rf_imp)

    lr_model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)               # baseline for comparison
    base_p = lr_model.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    metrics = {
        "backend": BACKEND,
        "architecture": {"input": len(feats), "hidden": list(hidden), "dropout": dropout, "learning_rate": lr,
                         "output": "1 sigmoid unit", "optimizer": "Adam", "loss": "binary cross-entropy",
                         "batch_size": BATCH, "max_epochs": EPOCHS, "early_stopping_patience": PATIENCE},
        "epochs_run": len(hist["loss"]),
        "best_epoch": int(np.argmin(hist["val_loss"])) + 1,
        "test": {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
                 "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "auc": roc_auc,
                 "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "baseline_logreg_test": {"accuracy": accuracy_score(yte, base_p > .5), "auc": float(auc(*roc_curve(yte, base_p)[:2]))},
        # accuracies at the best epoch (the weights that were restored by early stopping)
        "final_train_acc": hist.get("accuracy", hist.get("acc"))[int(np.argmin(hist["val_loss"]))],
        "final_val_acc": hist.get("val_accuracy", hist.get("val_acc"))[int(np.argmin(hist["val_loss"]))],
        "top_features_mlp": imp.head(5).round(4).to_dict(),
        "top_features_rf": rf_imp.head(5).round(4).to_dict(),
        "tuning_top3": tuning.head(3).to_dict(orient="records"),
        "tuning_n_configs": int(len(tuning)),
        "classification_report": classification_report(yte, pred, target_names=CLASS_NAMES, output_dict=True),
        "model_summary": model.summary_text(),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("model_summary", "classification_report")}, indent=2, default=float))


if __name__ == "__main__":
    main()
