"""Features: the cleaned store-day table in, the 34 model inputs out.

The helpers and checks are the notebook's, unchanged. Three things are learned from the training data and saved
with the model, so the API builds exactly the same features: the national holidays, the category lists, and the store list.
"""

import logging

import numpy as np
import pandas as pd

from .clean import set_missing
from .config import COMPETITION_CAP, HOLIDAY_CAP, HORIZON, LAGS, ROLLING_MIN_DAYS, ROLLING_WINDOW

logger = logging.getLogger(__name__)


# ---------- Helpers ----------

def add_lag(df, column, days, key, date_col, horizon):
    """Return a copy of df with `<column>_lag_<days>`: the value of `column` from `days` calendar days earlier, for the same `key`.

    The value is looked up by (key, date - days), so a missing day gives NaN and never pulls in another day.
    Raises ValueError if `days` is shorter than `horizon` (we would not know that value yet), or if a
    (key, date) pair appears twice. Safe to re-run: the lag column is rebuilt from `column` each time.
    """
    if days < horizon:
        raise ValueError(f"lag of {days} days is shorter than the {horizon}-day horizon")
    history = df.set_index([key, date_col])[column]
    if not history.index.is_unique:
        raise ValueError(f"{key} and {date_col} do not identify a row uniquely")
    earlier = pd.MultiIndex.from_arrays([df[key], df[date_col] - pd.Timedelta(days=days)])
    return df.assign(**{f"{column}_lag_{days}": history.reindex(earlier).to_numpy()})


def add_rolling_mean(df, column, window, key, date_col, horizon, min_periods):
    """Return a copy of df with `<column>_mean_<window>d`: the mean of `column` over the `window` calendar days
    that end `horizon` days before each row's date, for the same `key`.

    The days come from a complete calendar (one row per date for every key), so a window is always `window`
    calendar days, and missing days are skipped. A mean needs at least `min_periods` days, else it is NaN.
    Raises ValueError if a (key, date) pair appears twice. Safe to re-run: the column is rebuilt each time.
    """
    wide = df.pivot(index=date_col, columns=key, values=column)
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    rolled = wide.rolling(window, min_periods=min_periods).mean().shift(horizon)
    rows = pd.MultiIndex.from_arrays([df[date_col], df[key]])
    return df.assign(**{f"{column}_mean_{window}d": rolled.stack().reindex(rows).to_numpy()})


def add_holiday_distance(df, date_col, holiday_dates, cap):
    """Return a copy of df with `days_to_holiday` and `days_since_holiday`: calendar days to the next and since
    the last date in `holiday_dates` (0 on a holiday itself), capped at `cap`.

    If no holiday lies after (or before) a date in `holiday_dates`, the value is `cap`: we assume the next
    one is at least `cap` days away. Safe to re-run: both columns are rebuilt from the dates each time.
    """
    holidays = pd.DatetimeIndex(sorted(set(holiday_dates)))
    dates = pd.DatetimeIndex(df[date_col])
    next_i = holidays.searchsorted(dates, side="left")
    prev_i = holidays.searchsorted(dates, side="right") - 1
    to_next = np.where(next_i < len(holidays), (holidays[np.minimum(next_i, len(holidays) - 1)] - dates).days, cap)
    since_last = np.where(prev_i >= 0, (dates - holidays[np.maximum(prev_i, 0)]).days, cap)
    return df.assign(days_to_holiday=np.minimum(to_next, cap), days_since_holiday=np.minimum(since_last, cap))


def add_months_since(df, date_col, since_col, new_col, cap):
    """Return a copy of df with `new_col`: whole calendar months from `since_col` to `date_col`.

    The count is floored at 0 (the event has not happened yet, or happens this month) and capped at `cap`.
    A missing `since_col` stays missing. Safe to re-run: the column is rebuilt from the two dates each time.
    """
    months = (df[date_col].dt.year - df[since_col].dt.year) * 12 + (df[date_col].dt.month - df[since_col].dt.month)
    return df.assign(**{new_col: months.clip(lower=0, upper=cap)})


def add_promo2_flags(df, date_col, since_col, interval_col):
    """Return a copy of df with two 0/1 flags for the Promo2 programme.

    `promo2_active` is 1 on and after the date the store joined (`since_col`). Stores that never joined have
    no date, so they stay 0. `promo2_round_month` is 1 when the store is active and the month is one of the
    months in `interval_col` (for example "Jan,Apr,Jul,Oct"), when a new round starts.
    Safe to re-run: both columns are rebuilt from the dates each time.
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months_of = {s: {month_names.index(m) + 1 for m in s.split(",") if m in month_names} for s in df[interval_col].unique()}
    in_round_month = np.array([m in months_of[s] for m, s in zip(df[date_col].dt.month, df[interval_col])])
    active = df[date_col] >= df[since_col]
    return df.assign(promo2_active=active.astype(int), promo2_round_month=(active & in_round_month).astype(int))


def add_one_hot(df, categories):
    """Return a copy of df with one 0/1 column per category value, named `<column>_<value>`.

    `categories` maps each column to the full list of values it can take, so any slice of the data (a fold,
    or a single row) gets the same columns. The original columns are kept.
    Raises ValueError if a column has a value that is not in its list.
    Safe to re-run: the 0/1 columns are rebuilt from the original columns each time.
    """
    for col, values in categories.items():
        unknown = set(df[col].unique()) - set(values)
        if unknown:
            raise ValueError(f"{col} has values that are not in the list: {sorted(unknown)}")
    new = {f"{col}_{value}": (df[col] == value).astype(int) for col, values in categories.items() for value in values}
    return df.assign(**new)


# ---------- Learned from the training data (saved with the model) ----------

def find_national_holidays(fct, share):
    """Return the dates where at least `share` of the stores have a state holiday flagged."""
    share_off = fct.assign(off=fct["state_holiday"] != "none").groupby("date")["off"].mean()
    holidays = share_off[share_off >= share].index
    logger.info(f"national holidays: {len(holidays)} (flagged in at least {share:.0%} of stores)")
    return holidays


def category_values(fct, columns):
    """Return {column: sorted list of its values}, the full lists the one-hot columns are built from."""
    return {col: sorted(fct[col].unique()) for col in columns}


def feature_list(categories):
    """Return the 34 model inputs, in the notebook's order (six families)."""
    families = {
        "store and day": ["store_id", "day_of_week", "promo", "school_holiday", "competition_distance", "promo2"],
        "sales history": [*[f"sales_lag_{days}" for days in LAGS], "sales_lag_mean", f"sales_mean_{ROLLING_WINDOW}d"],
        "promo of the lag days": [f"promo_lag_{days}" for days in LAGS],
        "calendar and holiday": ["month", "day_of_month", "days_to_holiday", "days_since_holiday"],
        "store features": ["competition_months_open", "promo2_active", "promo2_round_month"],
        "encoded categories": [f"{col}_{value}" for col, values in categories.items() for value in values],
    }
    return [col for cols in families.values() for col in cols]


# ---------- Pipeline ----------

def build_features(fct, national_holidays, categories):
    """Return a copy of a cleaned store-day table with every feature column, built in the notebook's order.

    Works on the full history (training) or on one store's recent days plus the days to forecast (the API).
    Sales is set to missing on closed days first, so a lag never reads a closure as zero demand.
    """
    df = set_missing(fct, ["sales"], fct["open"] == 0)
    for days in LAGS:
        df = add_lag(df, "sales", days, key="store_id", date_col="date", horizon=HORIZON)
    for days in LAGS:
        df = add_lag(df, "promo", days, key="store_id", date_col="date", horizon=HORIZON)
    df = df.assign(sales_lag_mean=df[[f"sales_lag_{days}" for days in LAGS]].mean(axis=1))
    df = add_rolling_mean(df, "sales", ROLLING_WINDOW, key="store_id", date_col="date",
                          horizon=HORIZON, min_periods=ROLLING_MIN_DAYS)
    df = add_holiday_distance(df, "date", national_holidays, HOLIDAY_CAP)
    df = df.assign(month=df["date"].dt.month, day_of_month=df["date"].dt.day)
    df = add_months_since(df, "date", "competition_open_since", "competition_months_open", COMPETITION_CAP)
    df = add_promo2_flags(df, "date", "promo2_since", "promo_interval")
    return add_one_hot(df, categories)


def model_table(features_df, features, store_ids):
    """Return the open days only, with `date`, `sales`, and the features, and a fresh row index.

    `store_id` becomes a category over the fixed list `store_ids`, so training and the API code stores
    the same way (LightGBM reads it as a category).
    """
    table = features_df.loc[features_df["open"] == 1, ["date", "sales", *features]].reset_index(drop=True)
    return table.assign(store_id=pd.Categorical(table["store_id"], categories=store_ids))


# ---------- Checks ----------

def feature_checks(df, raw, features, categories):
    """Checks for the feature table. `raw` is the cleaned table the features were built from."""
    may_be_empty = {c for c in features if "_lag_" in c or "_mean_" in c or c == "competition_months_open"}
    one_hot_groups = [[f"{col}_{value}" for value in values] for col, values in categories.items()]
    return [
        ("same rows as the cleaned table", len(df) == len(raw)),
        ("no duplicate store-date rows", not df.duplicated(["store_id", "date"]).any()),
        ("sales is missing exactly on closed days", (df["sales"].isna() == (df["open"] == 0)).all()),
        ("the target and same-day customers are not features", not ({"sales", "customers"} & set(features))),
        ("only lags, means, and months open can be empty", set(df[features].columns[df[features].isna().any()]) <= may_be_empty),
        ("each one-hot group has exactly one 1 per row", all((df[cols].sum(axis=1) == 1).all() for cols in one_hot_groups)),
    ]


def leak_checks(df, lags, horizon, window, min_periods):
    """Checks that no sales feature looks at the last `horizon - 1` days before its row.

    Pretends that sales from a cutoff date onward were different, rebuilds the sales features, and compares.
    """
    cutoff = df["date"].median()
    altered = df.assign(sales=df["sales"].where(df["date"] < cutoff, 0.0))
    for days in lags:
        altered = add_lag(altered, "sales", days, "store_id", "date", horizon)
    altered = add_rolling_mean(altered, "sales", window, "store_id", "date", horizon, min_periods)
    lag_cols = [f"sales_lag_{days}" for days in lags]
    altered = altered.assign(sales_lag_mean=altered[lag_cols].mean(axis=1))
    sales_features = [*lag_cols, "sales_lag_mean", f"sales_mean_{window}d"]

    before = df["date"] < cutoff + pd.Timedelta(days=horizon)
    at_horizon = df["date"] == cutoff + pd.Timedelta(days=horizon)
    return [
        (f"every lag is at least {horizon} days", min(lags) >= horizon),
        (f"sales features ignore sales from the {horizon - 1} days before a row",
         df.loc[before, sales_features].equals(altered.loc[before, sales_features])),
        (f"...but do react to sales exactly {horizon} days back (the test has teeth)",
         not df.loc[at_horizon, sales_features].equals(altered.loc[at_horizon, sales_features])),
    ]
