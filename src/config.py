"""Settings shared by training and the API: paths, forecast settings, cleaning rules, and the model."""

from pathlib import Path

# Paths, built from the repo root, so the scripts work from any folder and on any OS
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TRAIN_CSV = RAW_DIR / "train.csv"
STORE_CSV = RAW_DIR / "store.csv"
MODEL_PATH = ROOT / "models" / "forecast_bundle.pkl"

# Forecast settings
RANDOM_SEED = 1           # reproducible randomness
HORIZON = 7               # forecast days ahead
HOLDOUT_DAYS = 42         # final holdout: the last 6 weeks
LAGS = [7, 14, 21, 28]    # sales lags in days, all >= HORIZON
ROLLING_WINDOW = 28       # days in the rolling sales mean
ROLLING_MIN_DAYS = 7      # open days a rolling mean needs
HOLIDAY_CAP = 14          # days to or since a holiday are capped here
NATIONAL_SHARE = 0.9      # a holiday is national when this share of stores has it flagged
COMPETITION_CAP = 24      # months since a competitor opened are capped here
HISTORY_DAYS = ROLLING_WINDOW + HORIZON  # days of sales the API keeps per store (enough for every lag and the rolling mean)
CATEGORY_COLUMNS = ["store_type", "assortment", "state_holiday"]  # text columns we one-hot encode

# Cleaning rules (the same as the notebook's cleaning log)
HOLIDAY_LABELS = {"0": "none", "a": "public", "b": "easter", "c": "christmas"}
ASSORTMENT_LABELS = {"a": "basic", "b": "extra", "c": "extended"}
PROMO_INTERVAL_FIXES = {"Sept": "Sep"}
MIN_PLAUSIBLE_YEAR = 1950  # competition years before this are treated as placeholders

# Model: the tuned LightGBM (Optuna trial 8 in the notebook)
MODEL_NAME = "lightgbm_tuned"
LGBM_PARAMS = {
    "num_leaves": 233,
    "min_child_samples": 51,
    "colsample_bytree": 0.785,
    "n_estimators": 357,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}
