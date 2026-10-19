# Breast Cancer Wisconsin – EDA, Visualization & MLP Classifier

Course: **U21ADP05 – Exploratory Data Analysis and Visualization** · Assignment II (Case Study)
KPR Institute of Engineering and Technology · Dept. of AI & Data Science

**Student:** YOUR NAME · **Roll No.:** YOUR ROLL NO

## Problem
Classify breast masses as **malignant** or **benign** from 30 numeric nuclear features of digitized fine-needle-aspirate images,
and use EDA/visualization to understand which features matter and why.

## Dataset
UCI Breast Cancer Wisconsin (Diagnostic): 569 rows, 30 features, 212 malignant / 357 benign. See [`data/README.md`](data/README.md).

## What this project does
1. **EDA & preprocessing** – missing values, duplicates, IQR outliers, clipping, standardisation, stratified 70/15/15 split.
2. **Visualization (9 charts, 7 stages)** – class distribution, histograms + normal fit, boxplots, correlation heatmap,
   scatter, pairplot, Q-Q plots, mutual information, before/after scaling (+ optional interactive Plotly chart).
3. **Deep learning** – MLP (Keras/TensorFlow) with a 12-configuration grid search, permutation importance, Random-Forest comparison.
4. **Evaluation** – loss/accuracy vs epoch, confusion matrix, ROC-AUC, logistic-regression baseline.

## Project structure
```
breast-cancer-eda-mlp/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                  # dataset links
│   ├── breast_cancer_wisconsin.csv
│   └── processed/                 # train.csv, val.csv, test.csv
├── src/
│   ├── config.py                  # paths, seed
│   ├── eda_visualization.py       # Steps 1-3
│   ├── model_mlp.py               # Steps 4-5
│   └── numpy_mlp.py               # fallback if TensorFlow is missing
├── results/
│   ├── figures/                   # all PNG charts
│   ├── metrics.json, eda_info.json, hyperparameter_tuning.csv, summary_statistics.csv
├── report/
│   ├── build_report.py            # builds the Word report from results/
│   ├── Project_Report.docx
│   └── Project_Report.pdf
└── docs/GITHUB_GUIDE.md
```

## How to run
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/eda_visualization.py     # EDA + figures + processed data
python src/model_mlp.py             # train, tune, evaluate
python report/build_report.py       # regenerate the Word report (edit student details inside first)
```

## Results
See `results/metrics.json` and the report. Test accuracy, F1 and ROC-AUC are written there after each run.

## Tools
Python · pandas · NumPy · SciPy · scikit-learn · Matplotlib · Seaborn · TensorFlow/Keras · python-docx
