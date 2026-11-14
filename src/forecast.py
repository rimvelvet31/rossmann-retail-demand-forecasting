"""Forecast: one store and its plan for the next 7 days in, 7 daily sales forecasts out.

The features are built with the same build_features() as training, from the store's recent history saved in
the bundle plus the 7 new days, so training and serving cannot drift apart.
"""

import logging
import pickle

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from .config import HORIZON
from .features import build_features, model_table

logger = logging.getLogger(__name__)

PLAN_COLUMNS = ["date", "open", "promo", "state_holiday", "school_holiday"]  # what the caller knows in advance


def load_bundle(path):
    """Load the bundle saved by train.py, and warn if the installed library versions differ from training.

    Only load bundles you built yourself: pickle can run code from the file it loads.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python train.py")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    installed = {"lightgbm": lightgbm.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    for lib, version in bundle["versions"].items():
        if installed[lib] != version:
            logger.warning(f"{lib} {installed[lib]} is installed, but the bundle was saved with {version}")
    logger.info(f"loaded {bundle['model_name']}, trained until {bundle['trained_until'].date()}")
    return bundle


def forecast_days(last_date, horizon):
    """Return the `horizon` dates right after `last_date`: the days the saved history can forecast."""
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


def build_forecast_rows(history, plan, national_holidays, categories):
    """Return the feature rows for the days in `plan`, built from one store's `history` plus those days.

    `history` holds the store's recent cleaned rows (all fct_store_day columns). `plan` has one row per day
    with PLAN_COLUMNS. The store's details (type, competition, Promo2) are copied from its last history row.
    """
    last = history.iloc[[-1]]
    future = last.loc[last.index.repeat(len(plan))].reset_index(drop=True).assign(
        date=plan["date"].to_numpy(),
        day_of_week=(plan["date"].dt.day_of_week + 1).to_numpy(),
        open=plan["open"].to_numpy(),
        promo=plan["promo"].to_numpy(),
        state_holiday=plan["state_holiday"].to_numpy(),
        school_holiday=plan["school_holiday"].to_numpy(),
        sales=np.nan,
        customers=np.nan,
    )
    features_df = build_features(pd.concat([history, future], ignore_index=True), national_holidays, categories)
    return features_df[features_df["date"] >= plan["date"].min()].reset_index(drop=True)


def forecast(bundle, store_id, plan):
    """Return a table with `date`, `open`, and `sales_forecast` for one store's next HORIZON days.

    `plan` has one row per day with PLAN_COLUMNS, and its dates must be the HORIZON days right after the saved
    history ends. Closed days get 0 without calling the model (sales is 0 by schedule, not demand).
    Raises ValueError for an unknown store or for the wrong dates.
    """
    if store_id not in bundle["store_ids"]:
        raise ValueError(f"unknown store_id {store_id}")
    plan = plan.assign(date=pd.to_datetime(plan["date"])).sort_values("date").reset_index(drop=True)
    expected = forecast_days(bundle["trained_until"], HORIZON)
    if list(plan["date"]) != list(expected):
        raise ValueError(f"the plan must have one row per day from {expected[0].date()} to {expected[-1].date()}")

    history = bundle["history"][bundle["history"]["store_id"] == store_id]
    rows = build_forecast_rows(history, plan, bundle["national_holidays"], bundle["categories"])
    open_rows = model_table(rows, bundle["features"], bundle["store_ids"])
    predicted = pd.Series(dtype=float)
    if len(open_rows):
        values = bundle["model"].predict(open_rows[bundle["features"]])
        predicted = pd.Series(values, index=open_rows["date"]).clip(lower=0)
    return rows[["date", "open"]].assign(sales_forecast=rows["date"].map(predicted).fillna(0.0).round(2))
