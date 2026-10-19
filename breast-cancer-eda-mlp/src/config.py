"""Central configuration so every script uses the same paths and seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
for _d in (DATA_DIR, PROC_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
RAW_CSV = DATA_DIR / "breast_cancer_wisconsin.csv"
TARGET = "diagnosis"          # 0 = Malignant, 1 = Benign (UCI WDBC coding used by scikit-learn)
CLASS_NAMES = ["Malignant", "Benign"]
