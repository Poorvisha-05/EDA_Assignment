"""
Builds report/Project_Report.docx from the files produced by the pipeline
(results/eda_info.json, results/metrics.json, results/figures/*.png, src/*.py).

1) python src/eda_visualization.py
2) python src/model_mlp.py
3) EDIT the STUDENT DETAILS block below
4) python report/build_report.py
"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# ============================ STUDENT DETAILS (EDIT THESE) ============================
STUDENT_NAME = "YOUR NAME"
ROLL_NO = "YOUR ROLL NO"
DEPARTMENT = "Artificial Intelligence and Data Science"
YEAR_SEM = "III Year / V Semester (Batch 2023-2027)"
INSTITUTE = "KPR Institute of Engineering and Technology"
COURSE = "U21ADP05 - Exploratory Data Analysis and Visualization"
COORDINATOR = "Mr. Rushikesh Kadam"
SUBMISSION_DATE = "DD/MM/2026"
GITHUB_URL = "https://github.com/<your-username>/breast-cancer-eda-mlp"
# ======================================================================================

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
eda = json.loads((ROOT / "results" / "eda_info.json").read_text())
met = json.loads((ROOT / "results" / "metrics.json").read_text())
stats_df = pd.read_csv(ROOT / "results" / "summary_statistics.csv", index_col=0)
tuning = pd.read_csv(ROOT / "results" / "hyperparameter_tuning.csv")
T = met["test"]
ARCH = met["architecture"]
INTERACTIVE = (ROOT / "results" / "interactive_scatter.html").exists()

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)      # A4
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Inches(1))
sec.top_margin = sec.bottom_margin = Inches(1)

# ---------- styles
st = doc.styles["Normal"]
st.font.name, st.font.size = "Times New Roman", Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.25
for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
    h = doc.styles[name]
    h.font.name, h.font.size, h.font.bold = "Times New Roman", Pt(size), True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(14 if name == "Heading 1" else 10), Pt(6)
    h.paragraph_format.keep_with_next = True

fig_no = [0]
tab_no = [0]


def para(text="", bold=False, italic=False, align=None, size=None, style=None):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if align == "j":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def body(text):
    return para(text, align="j")


def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def table(headers, rows, widths=None, caption=None, size=10):
    if caption:
        tab_no[0] += 1
        c = para(f"Table {tab_no[0]}: {caption}", bold=True, align="c", size=10.5)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size); r.font.color.rgb = RGBColor(255, 255, 255)
        shade(cell, "1F3A5F")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2); p.paragraph_format.line_spacing = 1.0
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(fname, caption, width=5.6):
    fig_no[0] += 1
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / fname), width=Inches(width))
    para(f"Figure {fig_no[0]}: {caption}", italic=True, align="c", size=10.5)


def viz(title, fname, what, why, insight, width=5.0):
    doc.add_heading(title, level=3)
    figure(fname, title, width)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run("What it shows: ").bold = True; p.add_run(what)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run("Why it was created: ").bold = True; p.add_run(why)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run("Insights obtained: ").bold = True; p.add_run(insight)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def add_page_number(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        run._r.append(el)


add_page_number(sec)

# =============================== 1. TITLE PAGE ===============================
for _ in range(2):
    para()
para(INSTITUTE, bold=True, align="c", size=16)
para(f"Department of {DEPARTMENT}", bold=True, align="c", size=14)
para()
para(COURSE, align="c", size=13)
para("Assignment II - Case Study", align="c", size=13)
para()
para()
para("Exploratory Data Analysis and Visualization of the", bold=True, align="c", size=20)
para("Breast Cancer Wisconsin (Diagnostic) Dataset", bold=True, align="c", size=20)
para("with a Deep Learning (MLP) Classifier", bold=True, align="c", size=16)
para(); para()
table(["Item", "Details"], [
    ["Student Name", STUDENT_NAME], ["Roll No.", ROLL_NO], ["Department", DEPARTMENT],
    ["Year / Semester", YEAR_SEM], ["Course Coordinator", COORDINATOR],
    ["Date of Submission", SUBMISSION_DATE], ["GitHub Repository", GITHUB_URL]], widths=[2.0, 4.2], size=11)
page_break()

# =============================== 2. ABSTRACT ===============================
doc.add_heading("Abstract", level=1)
abstract = (
    f"Early and accurate classification of breast masses as malignant or benign supports timely clinical decisions. "
    f"This case study applies exploratory data analysis (EDA) and visualization to the Breast Cancer Wisconsin "
    f"(Diagnostic) dataset, containing {eda['rows']} patients and {eda['n_features']} numerical features computed from "
    f"digitized fine-needle-aspirate images. The data were checked for missing values, duplicates and outliers, "
    f"standardized, and split into train, validation and test sets. Nine visualizations covering all seven visualization "
    f"stages revealed strong class separation and heavy multicollinearity among size-related features. A multilayer "
    f"perceptron with hyperparameter tuning achieved {T['accuracy']*100:.1f}% test accuracy and an AUC of {T['auc']:.3f}. "
    f"Size and concavity features proved most informative, and a logistic regression baseline performed comparably.")
body(abstract)
ABSTRACT_WORDS = len(abstract.split())
para(f"(Word count: {ABSTRACT_WORDS})", italic=True, size=10)
para("Keywords: EDA, data visualization, breast cancer, multilayer perceptron, ROC-AUC", italic=True)

# =============================== 3. INTRODUCTION ===============================
doc.add_heading("1. Introduction and Objective", level=1)
body("Breast cancer is one of the most common cancers worldwide, and the outcome for a patient depends strongly on how early "
     "and how reliably a tumour is identified as malignant. Fine-needle aspiration (FNA) is a minimally invasive procedure in "
     "which cells are extracted from a breast mass and examined under a microscope. Digitizing the slide image and measuring the "
     "shape of each cell nucleus gives quantitative features that can be analysed statistically and used to train predictive models [1].")
body("Real-world data of this type is rarely ready for modelling: features are on different scales, strongly correlated, skewed, and "
     "contain extreme values. Exploratory data analysis and visualization are therefore the first and most important step, because "
     "they reveal what the data can and cannot support before any model is trained.")
doc.add_heading("1.1 Objectives", level=2)
for t in ["Select a real-world dataset with at least 15 features and describe its source, size and fields.",
          "Perform EDA: handle missing values, duplicates and outliers; transform, scale and split the data.",
          "Create at least five meaningful visualizations covering the seven stages of visualization, each explained.",
          "Analyse univariate distributions (discrete and continuous), and bivariate / multivariate relationships.",
          "Build, tune and evaluate a deep learning model (MLP) and interpret loss, accuracy, confusion matrix, ROC and AUC.",
          "Draw practical conclusions and suggest future improvements."]:
    bullet(t)

# =============================== 4. DATASET ===============================
doc.add_heading("2. Dataset Description", level=1)
body(f"The Breast Cancer Wisconsin (Diagnostic) dataset (WDBC) was created by Wolberg, Street and Mangasarian at the University of "
     f"Wisconsin and is hosted in the UCI Machine Learning Repository [1], [2]. It is also bundled with scikit-learn [3], which was "
     f"used to load it. Features describe characteristics of cell nuclei in a digitized FNA image.")
table(["Property", "Value"], [
    ["Source", "UCI Machine Learning Repository / scikit-learn (load_breast_cancer)"],
    ["Records (patients)", eda["rows"]], ["Input features", eda["n_features"]],
    ["Target", "diagnosis: 0 = Malignant, 1 = Benign"],
    ["Class counts", f"Malignant = {eda['class_counts']['Malignant']} ({eda['class_counts']['Malignant']/eda['rows']*100:.1f}%), "
                     f"Benign = {eda['class_counts']['Benign']} ({eda['class_counts']['Benign']/eda['rows']*100:.1f}%)"],
    ["Feature type", "All continuous (real-valued)"], ["Missing values", eda["missing_total"]], ["Duplicate rows", eda["duplicates"]]],
    widths=[1.8, 4.4], caption="Dataset overview")
body("Ten base measurements are computed for every nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, "
     "concave points, symmetry and fractal dimension. For each, three versions are stored - the mean, the standard error (SE) and the "
     "'worst' (mean of the three largest values) - giving 10 x 3 = 30 features.")
sel = ["mean radius", "mean texture", "mean area", "mean smoothness", "mean concavity", "worst area"]
table(["Feature", "Mean", "Std", "Min", "Median", "Max"],
      [[f, f"{stats_df.loc[f,'mean']:.3f}", f"{stats_df.loc[f,'std']:.3f}", f"{stats_df.loc[f,'min']:.3f}",
        f"{stats_df.loc[f,'50%']:.3f}", f"{stats_df.loc[f,'max']:.3f}"] for f in sel],
      widths=[1.8, 0.9, 0.9, 0.9, 0.9, 0.9], caption="Basic statistics of selected features (full table in results/summary_statistics.csv)")

# =============================== 5. EDA & PREPROCESSING ===============================
doc.add_heading("3. EDA and Preprocessing", level=1)
doc.add_heading("3.1 Methods used", level=2)
bullet(f" Dataset loaded into a pandas DataFrame ({eda['rows']} x {eda['columns']}); data types, shape and summary statistics inspected.", "Loading and inspection:")
bullet(f" No missing values ({eda['missing_total']}) and no duplicate rows ({eda['duplicates']}) were found, so no imputation or removal was needed.", "Missing values and duplicates:")
top5 = ", ".join(f"{k} ({v})" for k, v in eda["outliers_top5"].items())
bullet(f" The 1.5 x IQR rule flagged {eda['outliers_iqr_total']} values across all features; the most affected were {top5}. "
       f"Because such values correspond to genuinely large tumours (clinically meaningful), they were not deleted. Instead, extreme "
       f"values beyond 3 x IQR (fences computed on the training set only) were clipped - {eda['split_sizes']['extreme_values_clipped']} values - "
       f"to limit their influence on the neural network.", "Outliers:")
bullet(" The target is already numerically encoded (0/1), and all 30 predictors are numerical, so no one-hot encoding was required.", "Encoding:")
bullet(" Features were standardized to zero mean and unit variance with StandardScaler, fitted on the training set only to avoid data leakage.", "Scaling:")
sp = eda["split_sizes"]
bullet(f" A stratified 70/15/15 split gave {sp['train']} training, {sp['val']} validation and {sp['test']} test samples, preserving the class ratio in every subset.", "Splitting:")
doc.add_heading("3.2 Insights from EDA", level=2)
sk = ", ".join(f"{k} ({v})" for k, v in eda["skew_top3"].items())
bullet(f" The classes are moderately imbalanced (about 63% benign, 37% malignant), so accuracy alone is not enough - recall, precision and AUC are also reported.")
bullet(f" Several features are strongly right-skewed (e.g., {sk}); Shapiro-Wilk tests rejected normality (p < 0.05) for {eda['n_nonnormal_shapiro']} of 30 features.")
bullet(f" Welch t-tests found a significant difference between malignant and benign means (p < 0.05) for {eda['n_sig_ttest']} of 30 features, confirming that most features carry diagnostic information.")
pairs = "; ".join(f"{k} (r = {v})" for k, v in list(eda["top_corr_pairs"].items())[:3])
bullet(f" Size features are almost perfectly correlated - {pairs} - indicating severe multicollinearity.")

# =============================== 6. VISUALIZATION ===============================
doc.add_heading("4. Data Visualization", level=1)
body("The visualizations follow the seven stages of data visualization described by Fry [6]. Table 3 maps each stage to the work done in the project.")
table(["Stage", "Activity in this project", "Output"], [
    ["1. Acquire", "Loaded WDBC dataset from scikit-learn / UCI and cached as CSV", "data/breast_cancer_wisconsin.csv"],
    ["2. Parse", "Inspected structure, types, missing values, duplicates, summary statistics", "summary_statistics.csv; Tables 1-2"],
    ["3. Filter", "Removed duplicates (none), identified IQR outliers, kept clinically valid values", "Section 3.1"],
    ["4. Mine", "Skewness, Shapiro-Wilk, t-tests, correlation, mutual information", "Section 3.2; Figure 8"],
    ["5. Represent", "Selected chart types by data type (bar, histogram, box, heatmap, scatter, pair, Q-Q)", "Figures 1-8"],
    ["6. Refine", "Titles, labels, colour-blind-friendly class colours, before/after scaling comparison", "Figure 9"],
    ["7. Interact", ("Interactive Plotly scatter with hover, zoom and legend toggle" if INTERACTIVE
                     else "Interactive Plotly scatter (generated when plotly is installed)"), "results/interactive_scatter.html"]],
    widths=[1.0, 3.6, 1.7], caption="Mapping of the seven visualization stages", size=9.5)

cc = eda["class_counts"]
viz("Class distribution", "v1_class_distribution.png",
    "A bar chart and pie chart of the number of malignant and benign patients (discrete target variable).",
    "To check class balance before modelling, since imbalance affects the choice of evaluation metrics. The target follows a Bernoulli distribution for each patient and a Binomial distribution for counts.",
    f"There are {cc['Benign']} benign ({cc['Benign']/eda['rows']*100:.1f}%) and {cc['Malignant']} malignant ({cc['Malignant']/eda['rows']*100:.1f}%) cases - moderately imbalanced but not severely, so stratified splitting is sufficient.", 5.2)
viz("Histograms with fitted normal curve", "v2_histograms_normal_fit.png",
    "Density histograms with a KDE and a fitted normal curve for mean radius, mean texture, mean area and mean concavity.",
    "To study each continuous variable's distribution (univariate analysis) and compare it with the normal distribution.",
    "Mean radius and mean texture are roughly bell-shaped (with mild right skew), whereas mean area and mean concavity are clearly right-skewed with long tails, so the normal curve fits them poorly. This supports scaling and outlier control before neural-network training.")
viz("Boxplots by diagnosis", "v3_boxplots_by_class.png",
    "Boxplots of six 'mean' features, split by malignant and benign.",
    "To compare class medians and spreads and to spot outliers.",
    "Malignant tumours have clearly higher radius, perimeter, area, concavity and concave points; the boxes overlap little for concave points and perimeter, which makes these good discriminators. Texture overlaps more. Several points lie beyond the whiskers, i.e., statistical outliers.")
viz("Correlation heatmap", "v4_correlation_heatmap.png",
    "Pearson correlation coefficients between the ten 'mean' features.",
    "To detect multicollinearity and relationships among variables (multivariate analysis).",
    "Radius, perimeter and area are almost perfectly correlated (r above 0.98) because they are geometrically related; concavity, concave points and compactness form a second correlated group. Smoothness and fractal dimension are weakly related to size. Redundant features justify regularization (dropout) in the model.", 5.4)
viz("Scatterplot: radius vs concave points", "v5_scatter_radius_concavepoints.png",
    "Each patient plotted by mean radius (x) and mean concave points (y), coloured by diagnosis.",
    "To visualize bivariate class separation in two of the most discriminating features.",
    "The two classes form largely separate clusters: malignant cases sit at larger radius and higher concave-points values. Only a narrow overlapping zone exists, which explains why even simple models perform well on this dataset.", 4.8)
viz("Pairplot of the five most informative features", "v6_pairplot_top5.png",
    "Pairwise scatterplots and diagonal histograms of the top five features ranked by mutual information.",
    "To examine multivariate structure and class separation across several features at once.",
    "Worst perimeter, worst area and worst radius form near-linear relationships (redundant), and all five features separate the classes along the diagonal distributions.", 5.6)
viz("Q-Q plots", "v7_qq_plots.png",
    "Quantile-quantile plots of mean radius, mean area and mean smoothness against a normal distribution.",
    "To assess normality of continuous variables more precisely than a histogram.",
    "Points far from the red line in the upper tail of mean area show a right-skewed (approximately log-normal) distribution, while mean smoothness is closer to normal. Features that deviate from normality benefit from standardization.")
viz("Mutual information with the diagnosis", "v8_mutual_information.png",
    "Horizontal bar chart of the mutual information between each feature and the diagnosis.",
    "To rank features by how much information they carry about the target without assuming linearity (feature analysis).",
    f"The top features are {', '.join(eda['mi_top5'].keys())}. 'Worst' and concave-point features dominate, while several standard-error features and fractal-dimension features carry little information.", 4.8)
viz("Effect of standardization", "v9_before_after_scaling.png",
    "Boxplots of four features before and after z-score standardization.",
    "To refine the data representation and show why scaling is required (the 'Refine' stage).",
    "Before scaling, mean area (hundreds to thousands) dwarfs smoothness (about 0.1), which would dominate gradient updates. After standardization all features share a comparable scale.")

doc.add_heading("4.1 Discrete and continuous probability distributions", level=2)
bullet(" The diagnosis of each patient is a Bernoulli trial (benign probability about 0.63); the number of benign cases in a sample follows a Binomial distribution.", "Discrete:")
bullet(" Mean radius and mean texture are roughly normal (mild right skew); area, concavity and error features are right-skewed and closer to log-normal. Normal-curve overlays (Figure 2), Q-Q plots (Figure 7) and Shapiro-Wilk tests were used for identification.", "Continuous:")

# =============================== 7. MODEL ===============================
doc.add_heading("5. Deep Learning Model", level=1)
body(f"A Multilayer Perceptron (MLP) was chosen because the data is tabular with numerical features. Implementation: {met['backend']}.")
doc.add_heading("5.1 Architecture", level=2)
h = ARCH["hidden"]
table(["Layer", "Type", "Output units", "Activation / Regularization"], [
    ["Input", "Input", ARCH["input"], "Standardized features"],
    ["Hidden 1", "Dense + Dropout", h[0], f"ReLU, dropout = {ARCH['dropout']}"],
    ["Hidden 2", "Dense + Dropout", h[1], f"ReLU, dropout = {ARCH['dropout']}"],
    ["Output", "Dense", 1, "Sigmoid (probability of Benign)"]], widths=[1.1, 1.6, 1.2, 2.3], caption="MLP architecture")
doc.add_heading("5.2 Training parameters", level=2)
table(["Parameter", "Value"], [
    ["Optimizer", f"{ARCH['optimizer']} (learning rate = {ARCH['learning_rate']})"], ["Loss", ARCH["loss"]],
    ["Batch size", ARCH["batch_size"]], ["Maximum epochs", ARCH["max_epochs"]],
    ["Early stopping", f"monitor val_loss, patience = {ARCH['early_stopping_patience']}, restore best weights"],
    ["Epochs actually run / best epoch", f"{met['epochs_run']} / {met['best_epoch']}"],
    ["Random seed", 42]], widths=[2.2, 4.0], caption="Training parameters")
doc.add_heading("5.3 Hyperparameter tuning", level=2)
body(f"A basic grid search over hidden-layer sizes, dropout rate and learning rate ({met['tuning_n_configs']} configurations) was run. "
     f"Each model was trained on the training set and compared on the validation set; the test set was not used for tuning. "
     f"The configuration with the lowest validation loss was selected. Because the validation set is small ({sp['val']} samples), "
     f"differences of one or two samples in accuracy are not statistically meaningful.")
table(["Hidden units", "Dropout", "Learning rate", "Best val. loss", "Best val. acc.", "Epochs"],
      [[r.hidden_units, r.dropout, r.learning_rate, r.best_val_loss, r.best_val_acc, r.epochs_run] for r in tuning.head(6).itertuples()],
      widths=[1.2, 0.9, 1.1, 1.1, 1.1, 0.8], caption="Top 6 hyperparameter configurations (sorted by validation loss)")
doc.add_heading("5.4 Feature analysis and importance", level=2)
body("Feature importance was computed in two ways: (i) permutation importance of the trained MLP (the drop in validation accuracy when "
     "one feature is shuffled) and (ii) a Random Forest as an independent reference.")
figure("r5_feature_importance.png", "Feature importance - MLP permutation importance and Random Forest importance", 6.1)
body(f"Both methods place the size-related 'worst' features at the top (MLP: {', '.join(met['top_features_mlp'].keys())}; "
     f"Random Forest: {', '.join(met['top_features_rf'].keys())}), consistent with the mutual-information ranking in Figure 8. "
     f"Permutation importance is small and spread out for the MLP because correlated features share the same information - shuffling one is compensated by its correlated partners.")

# =============================== 8. RESULTS ===============================
doc.add_heading("6. Result Visualization and Interpretation", level=1)
figure("r1_loss_vs_epoch.png", "Loss vs Epoch", 4.9)
body(f"Training loss decreases steadily as the network learns. The validation loss reaches its minimum at epoch {met['best_epoch']} and then stops "
     f"improving, so early stopping (patience {ARCH['early_stopping_patience']}) halted training at epoch {met['epochs_run']} and restored the weights from epoch {met['best_epoch']}. "
     f"The validation curve is noisier than the training curve because the validation set is small ({sp['val']} samples) and the learning rate is {ARCH['learning_rate']}; "
     f"early stopping prevents the network from overfitting the training data.")
figure("r2_accuracy_vs_epoch.png", "Accuracy vs Epoch", 4.9)
body(f"Accuracy at the best epoch is {met['final_train_acc']*100:.1f}% (training) and {met['final_val_acc']*100:.1f}% (validation). "
     f"The model learns the task quickly because the classes are well separated (Figure 5).")
figure("r3_confusion_matrix.png", "Confusion matrix on the test set", 3.9)
body(f"Of {sp['test']} test patients, {T['tp']} benign and {T['tn']} malignant cases were classified correctly; {T['fp']} malignant case(s) were wrongly "
     f"predicted as benign (false positive for the Benign class) and {T['fn']} benign case(s) as malignant. In a clinical setting a malignant tumour predicted as benign is the most "
     f"serious error, so this count should be minimized, for example by lowering the decision threshold.")
figure("r4_roc_curve.png", "ROC curve and AUC on the test set", 4.4)
body(f"The ROC curve hugs the top-left corner and the area under the curve is {T['auc']:.3f}, meaning the model ranks a random benign case above a random malignant case with probability {T['auc']*100:.1f}%.")
table(["Metric", "MLP (test)", "Logistic regression baseline (test)"], [
    ["Accuracy", f"{T['accuracy']:.4f}", f"{met['baseline_logreg_test']['accuracy']:.4f}"],
    ["Precision (Benign)", f"{T['precision']:.4f}", "-"], ["Recall (Benign)", f"{T['recall']:.4f}", "-"],
    ["F1-score", f"{T['f1']:.4f}", "-"], ["ROC-AUC", f"{T['auc']:.4f}", f"{met['baseline_logreg_test']['auc']:.4f}"]],
    widths=[1.9, 1.7, 2.6], caption="Test-set performance summary")
body("The logistic regression baseline performs comparably to (or slightly better than) the MLP. This is a common finding on small, "
     "well-separated tabular datasets and shows that a deep model is not automatically superior; its main benefit here is as a learning exercise "
     "and as a base for larger data. With only 86 test samples, a difference of one patient changes accuracy by about 1.2 percentage points.")

# =============================== 9. CONCLUSION ===============================
doc.add_heading("7. Conclusion and Future Scope", level=1)
doc.add_heading("7.1 Conclusion", level=2)
bullet(" The dataset is clean (no missing values or duplicates), moderately imbalanced, with skewed features and many outliers corresponding to large tumours.", "What was learned:")
bullet(" Size and shape features (radius, perimeter, area, concavity, concave points - especially the 'worst' values) differ strongly between malignant and benign tumours; these features are highly inter-correlated.", "Key patterns:")
bullet(f" The tuned MLP reached {T['accuracy']*100:.1f}% accuracy, {T['f1']:.3f} F1 and {T['auc']:.3f} AUC on unseen test data, comparable to a logistic regression baseline.", "Model performance:")
bullet(" Small data size (569 rows) makes results sensitive to the particular split; heavy multicollinearity; the small validation set limits reliable tuning; and the model is not a medical-grade tool.", "Challenges and limitations:")
doc.add_heading("7.2 Future scope", level=2)
for t in ["Use k-fold cross-validation or repeated splits to obtain confidence intervals instead of a single split.",
          "Remove redundant features with PCA or correlation filtering and compare model performance.",
          "Tune hyperparameters with Keras Tuner or Optuna, and compare with gradient boosting (XGBoost / LightGBM).",
          "Use explainability tools such as SHAP or LIME to explain individual predictions to clinicians.",
          "Calibrate probabilities and choose a decision threshold that prioritises recall for malignant cases.",
          "Train a CNN directly on the FNA images (a bigger, external dataset) and validate on data from other hospitals."]:
    bullet(t)

# =============================== 10. REFERENCES ===============================
doc.add_heading("References", level=1)
refs = [
    "W. H. Wolberg, W. N. Street, and O. L. Mangasarian, \"Nuclear feature extraction for breast tumor diagnosis,\" in Proc. IS&T/SPIE Int. Symp. Electronic Imaging: Science and Technology, vol. 1905, 1993, pp. 861-870.",
    "D. Dua and C. Graff, \"UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Data Set,\" Univ. of California, Irvine, 2019. [Online]. Available: https://archive.ics.uci.edu/ml",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.",
    "D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" arXiv:1412.6980, 2014.",
    "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, \"Dropout: A simple way to prevent neural networks from overfitting,\" J. Mach. Learn. Res., vol. 15, pp. 1929-1958, 2014.",
    "B. Fry, Visualizing Data: Exploring and Explaining Data with the Processing Environment. Sebastopol, CA, USA: O'Reilly Media, 2007.",
    "M. Abadi et al., \"TensorFlow: Large-scale machine learning on heterogeneous systems,\" 2015. [Online]. Available: https://www.tensorflow.org"]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph(f"[{i}]  {r}"); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.4); p.paragraph_format.first_line_indent = Inches(-0.4)

# =============================== 11. APPENDIX ===============================
page_break()
doc.add_heading("Appendix - Source Code", level=1)
body(f"The complete project, including dataset, notebooks/scripts, results and this report, is available at: {GITHUB_URL}")
for fname in ("config.py", "eda_visualization.py", "numpy_mlp.py", "model_mlp.py"):
    doc.add_heading(f"src/{fname}", level=2)
    for line in (ROOT / "src" / fname).read_text().splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
        r = p.add_run(line if line else " ")
        r.font.name, r.font.size = "Courier New", Pt(7.5)
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")

out = ROOT / "report" / "Project_Report.docx"
doc.save(out)
print("Saved", out, "| abstract words:", ABSTRACT_WORDS)
