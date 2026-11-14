"""Train the forecast model: the raw Kaggle files in, models/forecast_bundle.pkl out.

Steps: 
1. Clean
2. Features 
3. Holdout check (fit before the holdout and score it once, as in the notebook)
4. Final fit on all open days 
5. Save the bundle the API loads.

Usage, from the repo root: uv run python train.py
"""

import logging
import pickle
import sys
import time

import lightgbm
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor

from src.checks import run_checks
from src.clean import clean, load_raw
from src.config import (
    CATEGORY_COLUMNS, HISTORY_DAYS, HOLDOUT_DAYS, HORIZON, LAGS, LGBM_PARAMS, MODEL_NAME, MODEL_PATH,
    NATIONAL_SHARE, ROLLING_MIN_DAYS, ROLLING_WINDOW, STORE_CSV, TRAIN_CSV,
)
from src.features import (
    build_features, category_values, feature_checks, feature_list, 
    find_national_holidays, leak_checks, model_table,
)

logger = logging.getLogger("train")


def score(actual, predicted):
    """Return the WAPE and MAE of `predicted` against `actual`. Raises ValueError if a prediction is missing."""
    if predicted.isna().any():
        raise ValueError(f"{predicted.isna().sum():,} predictions are missing")
    miss = (actual - predicted).abs()
    return {"wape": miss.sum() / actual.sum(), "mae": miss.mean()}


def fit_lgbm(table, features, label):
    """Fit the tuned LightGBM on `table`, log how long it took, and return it."""
    logger.info(f"fitting the {label} model on {len(table):,} rows...")
    fit_start = time.perf_counter()
    model = LGBMRegressor(**LGBM_PARAMS).fit(table[features], table["sales"])
    logger.info(f"{label} model fitted in {time.perf_counter() - fit_start:.0f} s")
    return model


def predict_sales(model, table, features):
    """Predict sales for `table`, raised to 0 where negative (sales cannot be negative)."""
    return pd.Series(model.predict(table[features]), index=table.index).clip(lower=0)


def main():
    start = time.perf_counter()

    # 1. Clean
    fct = clean(*load_raw(TRAIN_CSV, STORE_CSV))

    # 2. Learn what the features need from the data, then build and check the features
    national_holidays = find_national_holidays(fct, NATIONAL_SHARE)
    categories = category_values(fct, CATEGORY_COLUMNS)
    features = feature_list(categories)
    store_ids = sorted(int(s) for s in fct["store_id"].unique())
    features_df = build_features(fct, national_holidays, categories)
    run_checks(feature_checks(features_df, fct, features, categories), "features")
    run_checks(leak_checks(features_df, LAGS, HORIZON, ROLLING_WINDOW, ROLLING_MIN_DAYS), "leakage")
    table = model_table(features_df, features, store_ids)
    logger.info(f"model table: {len(table):,} open days, {len(features)} features")

    # 3. Holdout check: train on the days before the last 6 weeks, then score those 6 weeks once
    last_date = fct["date"].max()
    holdout_start = last_date - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    before = table[table["date"] < holdout_start]
    holdout = table[table["date"] >= holdout_start]
    holdout_model = fit_lgbm(before, features, "holdout")
    holdout_score = score(holdout["sales"], predict_sales(holdout_model, holdout, features))
    logger.info(f"holdout {holdout_start.date()} to {last_date.date()}: {len(holdout):,} rows, "
                f"WAPE {holdout_score['wape']:.4f}, MAE {holdout_score['mae']:.2f} EUR")

    # 4. Final fit on every open day, holdout included
    model = fit_lgbm(table, features, "final")
    logger.info(f"final model: {MODEL_NAME} fitted on {len(table):,} rows up to {last_date.date()}")

    # 5. Save everything the API needs in one file
    bundle = {
        "model": model,
        "model_name": MODEL_NAME,
        "trained_until": last_date,
        "features": features,
        "categories": categories,
        "store_ids": store_ids,
        "national_holidays": list(national_holidays),
        "history": fct[fct["date"] > last_date - pd.Timedelta(days=HISTORY_DAYS)].reset_index(drop=True),
        "holdout_score": holdout_score,
        "versions": {"lightgbm": lightgbm.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB, "
                f"{len(bundle['history']):,} history rows) in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        main()
    except FileNotFoundError as error:  # data not downloaded: one clear line, no traceback
        logger.error(error)
        sys.exit(1)
    except AssertionError as error:  # a data check failed: the FAIL lines above say which
        logger.error(f"stopped: {error}")
        sys.exit(1)
    except Exception:  # anything unexpected: the full traceback goes into the log
        logger.exception("training failed")
        sys.exit(1)
