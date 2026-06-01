from pathlib import Path

# PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "catboost_model.cbm"

# DATA

TARGET = "demand"

ID_COLUMN = "Index"

RANDOM_STATE = 80085

# CATBOOST

CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "random_seed": RANDOM_STATE,
    "verbose": 100,
}