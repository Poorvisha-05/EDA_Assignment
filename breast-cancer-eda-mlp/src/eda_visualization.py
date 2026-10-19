"""
Steps 1-3: Data acquisition, EDA, preprocessing and visualisation.

Maps to the 7 stages of data visualisation (Ben Fry):
  1 Acquire  2 Parse  3 Filter  4 Mine  5 Represent  6 Refine  7 Interact

Run:  python src/eda_visualization.py
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import CLASS_NAMES, FIG_DIR, PROC_DIR, RAW_CSV, RESULTS_DIR, SEED, TARGET

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Malignant": "#d62728", "Benign": "#1f77b4"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Stage 1: ACQUIRE
def acquire():
    """Load the UCI Breast Cancer Wisconsin (Diagnostic) dataset and cache it as CSV."""
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.rename(columns={"target": TARGET})
    df.to_csv(RAW_CSV, index=False)
    return df


# ---------------------------------------------------------------- Stage 2: PARSE
def parse(df):
    """Structure, types, missing values, duplicates, summary statistics."""
    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "n_features": int(df.shape[1] - 1),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "class_counts": {CLASS_NAMES[k]: int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
    }
    df.describe().T.round(3).to_csv(RESULTS_DIR / "summary_statistics.csv")
    return info


# ---------------------------------------------------------------- Stage 3: FILTER
def iqr_outliers(X):
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    return ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum()


def filter_data(df):
    """Remove duplicates; report outliers (kept - they are clinically valid)."""
    df = df.drop_duplicates().reset_index(drop=True)
    out = iqr_outliers(df.drop(columns=TARGET))
    return df, out


# ---------------------------------------------------------------- Stage 4: MINE
def mine(df):
    """Statistics: skewness, normality, correlation, mutual information, class t-tests."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    skew = X.skew().sort_values(ascending=False)
    shapiro = {c: stats.shapiro(X[c])[1] for c in X.columns}
    mi = pd.Series(mutual_info_classif(X, y, random_state=SEED), index=X.columns).sort_values(ascending=False)
    ttest = {c: stats.ttest_ind(X.loc[y == 0, c], X.loc[y == 1, c], equal_var=False)[1] for c in X.columns}
    corr = X.corr()
    target_corr = X.apply(lambda s: s.corr(y)).sort_values()
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(ascending=False)
    return {"skew": skew, "shapiro": shapiro, "mi": mi, "ttest": ttest, "corr": corr,
            "target_corr": target_corr, "top_pairs": pairs.head(5)}


# ---------------------------------------------------------------- Stage 5: REPRESENT
def represent(df, m):
    X, y = df.drop(columns=TARGET), df[TARGET]
    labels = y.map(dict(enumerate(CLASS_NAMES)))
    mean_cols = [c for c in X.columns if c.startswith("mean ")]

    # V1 - Class distribution (discrete: Bernoulli / Binomial)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    counts = labels.value_counts()[CLASS_NAMES]
    sns.barplot(x=counts.index, y=counts.values, palette=PALETTE, ax=ax[0])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 4, str(v), ha="center", fontweight="bold")
    ax[0].set(title="Class distribution (count)", xlabel="Diagnosis", ylabel="Number of patients")
    ax[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=[PALETTE[c] for c in counts.index], startangle=90)
    ax[1].set_title("Class proportion")
    save(fig, "v1_class_distribution.png")

    # V2 - Histograms + KDE + fitted normal curve (continuous distributions)
    sel = ["mean radius", "mean texture", "mean area", "mean concavity"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, c in zip(axes.ravel(), sel):
        sns.histplot(X[c], kde=True, stat="density", bins=30, color="#4c72b0", ax=ax)
        xs = np.linspace(X[c].min(), X[c].max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, X[c].mean(), X[c].std()), "r--", label="Normal fit")
        ax.set_title(f"{c}  (skew={X[c].skew():.2f})")
        ax.legend()
    fig.suptitle("Univariate distributions with fitted normal curve", y=1.02, fontsize=14)
    save(fig, "v2_histograms_normal_fit.png")

    # V3 - Boxplots by class (outliers + class separation)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), ["mean radius", "mean texture", "mean perimeter", "mean area", "mean concavity", "mean concave points"]):
        sns.boxplot(x=labels, y=X[c], order=CLASS_NAMES, palette=PALETTE, ax=ax)
        ax.set(title=c, xlabel="", ylabel="")
    fig.suptitle("Boxplots by diagnosis", y=1.02, fontsize=14)
    save(fig, "v3_boxplots_by_class.png")

    # V4 - Correlation heatmap (mean features)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(X[mean_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation heatmap - 'mean' features")
    save(fig, "v4_correlation_heatmap.png")

    # V5 - Scatterplot radius vs concave points coloured by class
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=X.assign(Diagnosis=labels), x="mean radius", y="mean concave points", hue="Diagnosis",
                    palette=PALETTE, alpha=.75, edgecolor="w", ax=ax)
    ax.set_title("Mean radius vs mean concave points")
    save(fig, "v5_scatter_radius_concavepoints.png")

    # V6 - Pairplot of 5 strongest features
    top5 = m["mi"].index[:5].tolist()
    g = sns.pairplot(X[top5].assign(Diagnosis=labels), hue="Diagnosis", palette=PALETTE, corner=True, plot_kws={"s": 18, "alpha": .7})
    g.figure.suptitle("Pairplot of the 5 most informative features", y=1.02)
    g.figure.savefig(FIG_DIR / "v6_pairplot_top5.png", dpi=130, bbox_inches="tight")
    plt.close(g.figure)

    # V7 - Q-Q plots (normality check for continuous distributions)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, c in zip(axes, ["mean radius", "mean area", "mean smoothness"]):
        stats.probplot(X[c], dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot: {c}")
    save(fig, "v7_qq_plots.png")

    # V8 - Mutual information / feature-target correlation (Mine -> Represent)
    fig, ax = plt.subplots(figsize=(8, 8))
    mi = m["mi"].sort_values()
    ax.barh(mi.index, mi.values, color="#2a9d8f")
    ax.set(title="Mutual information with diagnosis", xlabel="Mutual information")
    save(fig, "v8_mutual_information.png")


# ---------------------------------------------------------------- Stage 6: REFINE
def refine(df):
    """Show the effect of preprocessing: before / after scaling + outlier clipping."""
    X = df.drop(columns=TARGET)
    cols = ["mean radius", "mean area", "mean smoothness", "mean concavity"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=X[cols], ax=ax[0]); ax[0].set_title("Before scaling (different units)")
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)[cols]
    sns.boxplot(data=Xs, ax=ax[1]); ax[1].set_title("After standardisation (z-score)")
    save(fig, "v9_before_after_scaling.png")


# ---------------------------------------------------------------- Stage 7: INTERACT
def interact(df):
    """Interactive Plotly scatter (hover, zoom, legend toggle). Optional dependency."""
    try:
        import plotly.express as px
    except ImportError:
        print("plotly not installed -> skipping interactive chart (pip install plotly)")
        return False
    d = df.assign(Diagnosis=df[TARGET].map(dict(enumerate(CLASS_NAMES))))
    fig = px.scatter(d, x="mean radius", y="mean concave points", color="Diagnosis", size="mean area",
                     hover_data=["mean texture", "mean perimeter"],
                     color_discrete_map=PALETTE, title="Interactive: radius vs concave points (bubble = area)")
    fig.write_html(RESULTS_DIR / "interactive_scatter.html")
    return True


# ---------------------------------------------------------------- Preprocess + split
def preprocess_split(df):
    """Clip extreme outliers (3*IQR, fences from TRAIN only), standardise (fit on TRAIN only), 70/15/15 split."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=.50, stratify=y_tmp, random_state=SEED)

    q1, q3 = X_tr.quantile(.25), X_tr.quantile(.75)
    lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
    n_clipped = int(((X_tr < lo) | (X_tr > hi)).sum().sum())
    X_tr, X_val, X_te = (s.clip(lo, hi, axis=1) for s in (X_tr, X_val, X_te))

    sc = StandardScaler().fit(X_tr)
    out = {}
    for name, Xs, ys in (("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)):
        arr = pd.DataFrame(sc.transform(Xs), columns=X.columns, index=Xs.index).assign(**{TARGET: ys.values})
        arr.to_csv(PROC_DIR / f"{name}.csv", index=False)
        out[name] = int(len(arr))
    out["extreme_values_clipped"] = n_clipped
    return out


def main():
    df = acquire()                                    # 1 Acquire
    info = parse(df)                                  # 2 Parse
    df, out = filter_data(df)                         # 3 Filter
    m = mine(df)                                      # 4 Mine
    represent(df, m)                                  # 5 Represent
    refine(df)                                        # 6 Refine
    info["interactive_html"] = interact(df)           # 7 Interact
    info["split_sizes"] = preprocess_split(df)
    info["outliers_iqr_total"] = int(out.sum())
    info["outliers_top5"] = out.sort_values(ascending=False).head(5).astype(int).to_dict()
    info["skew_top3"] = m["skew"].head(3).round(2).to_dict()
    info["n_nonnormal_shapiro"] = int(sum(p < .05 for p in m["shapiro"].values()))
    info["n_sig_ttest"] = int(sum(p < .05 for p in m["ttest"].values()))
    info["mi_top5"] = m["mi"].head(5).round(3).to_dict()
    info["target_corr_top"] = {"most_negative": m["target_corr"].head(3).round(3).to_dict(),
                               "most_positive": m["target_corr"].tail(3).round(3).to_dict()}
    info["top_corr_pairs"] = {" | ".join(k): round(float(v), 3) for k, v in m["top_pairs"].items()}
    (RESULTS_DIR / "eda_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
