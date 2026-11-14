"""Cleaning: the two raw Kaggle tables in, one checked store-day table (fct_store_day) out.

The helpers and checks are the notebook's, unchanged. 

clean() applies the 12 fixes in the notebook's order.
"""

import logging
import re

import pandas as pd

from .config import ASSORTMENT_LABELS, HOLIDAY_LABELS, MIN_PLAUSIBLE_YEAR, PROMO_INTERVAL_FIXES

logger = logging.getLogger(__name__)


# ---------- Helpers ----------

def to_snake_case(name):
    """StateHoliday -> state_holiday, Promo2SinceWeek -> promo2_since_week."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def recode_categories(df, column, labels):
    """Return a copy of df with `column` recoded through the `labels` dict.

    Safe to re-run: if every value is already a label, the data is returned unchanged.
    Raises ValueError for unknown codes, or for a mix of old and new codes.
    Missing values stay missing.
    """
    codes = set(df[column].dropna().unique())
    if codes <= set(labels):
        return df.assign(**{column: df[column].map(labels)})
    if codes <= set(labels.values()):
        return df  # already recoded, nothing to do
    unexpected = codes - set(labels) - set(labels.values())
    if unexpected:
        raise ValueError(f"Unexpected codes in {column!r}: {sorted(unexpected)}")
    raise ValueError(f"{column!r} is partly recoded (a mix of old and new codes)")


def fix_spelling(df, column, corrections):
    """Return a copy of df with each wrong spelling in `corrections` fixed inside `column`.

    `corrections` maps wrong -> right, for example {"Sept": "Sep"}. It matches text
    inside a value, not only whole values, and the match is literal (no regex).
    Missing values stay missing.
    """
    fixed = df[column]
    for wrong, right in corrections.items():
        fixed = fixed.str.replace(wrong, right, regex=False)
    return df.assign(**{column: fixed})


def sort_rows(df, by):
    """Return a copy of df sorted by the columns in `by` (oldest first), with a fresh row index."""
    return df.sort_values(by).reset_index(drop=True)


def set_missing(df, columns, where):
    """Return a copy of df with `columns` set to missing on the rows where `where` is True."""
    fixed = df.copy()
    fixed.loc[where, columns] = float("nan")
    return fixed


def fill_missing(df, column, value):
    """Return a copy of df with missing values in `column` replaced by `value`."""
    return df.assign(**{column: df[column].fillna(value)})


def combine_year_month(df, new_col, year, month, day=1):
    """Return a copy of df with a year column and a month column combined into one date column `new_col`.

    `day` is a fixed day of the month. The source columns are dropped.
    A missing year or month gives a missing date.
    Safe to re-run: if `new_col` exists and the source columns are gone, df is returned unchanged.
    """
    source_cols = [year, month]
    if new_col in df.columns and not any(c in df.columns for c in source_cols):
        return df
    dates = pd.to_datetime({"year": df[year], "month": df[month], "day": day}, errors="coerce")
    return df.drop(columns=source_cols).assign(**{new_col: dates})


def combine_year_week(df, new_col, year, week, weekday=1):
    """Return a copy of df with a year and an ISO week combined into one date column `new_col`.

    ISO 8601 weeks (the German standard): week 1 is the week with the first Thursday of the year.
    `weekday` picks the day of that week, 1 = Monday ... 7 = Sunday. The source columns are dropped.
    A missing year or week gives a missing date.
    Safe to re-run: if `new_col` exists and the source columns are gone, df is returned unchanged.
    """
    source_cols = [year, week]
    if new_col in df.columns and not any(c in df.columns for c in source_cols):
        return df
    jan4 = pd.to_datetime({"year": df[year], "month": 1, "day": 4}, errors="coerce")  # always in ISO week 1
    week1_start = jan4 - pd.to_timedelta(jan4.dt.day_of_week, unit="D")
    dates = week1_start + pd.to_timedelta((df[week] - 1) * 7 + (weekday - 1), unit="D")
    return df.drop(columns=source_cols).assign(**{new_col: dates})


def drop_rows_where(df, condition):
    """Return a copy of df without the rows where `condition(df)` is True, with a fresh row index.

    `condition` is a function that takes the DataFrame and returns True or False per row, so it is
    always checked against the current data.
    """
    return df.loc[~condition(df)].reset_index(drop=True)


def is_zero_sales_open_day(df):
    return (df["open"] == 1) & (df["sales"] == 0)


def merge_checked(left, right, on):
    """Left-join `right` onto `left` on the key `on`, and check that nothing was gained or lost.

    Raises ValueError if `right` has duplicate keys (many-to-one join), if a key in `left` has no
    match in `right`, or if the row count changes.
    """
    merged = left.merge(right, on=on, how="left", validate="many_to_one", indicator=True)
    unmatched = (merged["_merge"] != "both").sum()
    if unmatched:
        raise ValueError(f"{unmatched:,} rows in `left` have no match in `right` on {on!r}")
    if len(merged) != len(left):
        raise ValueError(f"Row count changed: {len(left):,} -> {len(merged):,}")
    return merged.drop(columns="_merge")


# ---------- Checks ----------

def run_checks(checks):
    """Log PASS or FAIL for each (name, passed) pair, and raise if any check failed."""
    for name, passed in checks:
        if passed:
            logger.info(f"PASS  {name}")
        else:
            logger.error(f"FAIL  {name}")
    failed = [name for name, passed in checks if not passed]
    if failed:
        raise AssertionError(f"{len(failed)} check(s) failed: {failed}")


def train_checks(df, raw, holiday_labels):
    """Checks for the cleaned train table. `raw` is the untouched train table."""
    dropped = ((raw["Open"] == 1) & (raw["Sales"] == 0)).sum()
    is_open = df["open"] == 1
    return [
        ("no duplicate store-date rows", not df.duplicated(["store_id", "date"]).any()),
        ("sorted by store_id, then date", pd.MultiIndex.from_frame(df[["store_id", "date"]]).is_monotonic_increasing),
        ("no missing values", not df.isna().any().any()),
        ("day_of_week matches the date", ((df["date"].dt.day_of_week + 1) == df["day_of_week"]).all()),
        ("closed days have zero sales and zero customers", df.loc[~is_open, ["sales", "customers"]].eq(0).all().all()),
        ("open days have sales and customers above zero", (df.loc[is_open, ["sales", "customers"]] > 0).all().all()),
        ("state_holiday has only the known labels", df["state_holiday"].isin(holiday_labels.values()).all()),
        ("row count = raw rows minus the dropped zero-sales days", len(df) == len(raw) - dropped),
    ]


def store_checks(df, raw):
    """Checks for the cleaned store table. `raw` is the untouched store table."""
    missing_ok = {"competition_open_since", "promo2_since"}
    return [
        ("one row per store, same count as raw", df["store_id"].is_unique and len(df) == len(raw)),
        ("only the two date columns have missing values", set(df.columns[df.isna().any()]) <= missing_ok),
        ("promo2_since is missing exactly when promo2 = 0", (df["promo2_since"].isna() == (df["promo2"] == 0)).all()),
        ("competition_distance is positive", (df["competition_distance"] > 0).all()),
    ]


def merged_checks(df, left, right, on):
    """Checks for the merged table. `left` and `right` are the two tables that went into it."""
    date_columns = {"competition_open_since", "promo2_since"}
    return [
        ("row count is the same as before the merge", len(df) == len(left)),
        ("every store in the store table has sales rows", set(right[on]) <= set(df[on])),
        ("column count = left + right - the key", df.shape[1] == left.shape[1] + right.shape[1] - 1),
        ("only the two date columns have missing values", set(df.columns[df.isna().any()]) <= date_columns),
    ]


# ---------- Pipeline ----------

def load_raw(train_csv, store_csv):
    """Read the two raw Kaggle files. Raises FileNotFoundError with the download command if one is missing."""
    for path in (train_csv, store_csv):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/download_data.py")
    train = pd.read_csv(train_csv, dtype={"StateHoliday": "str"}, parse_dates=["Date"])
    store = pd.read_csv(store_csv)
    logger.info(f"loaded {train_csv.name}: {len(train):,} rows, {store_csv.name}: {len(store):,} rows")
    return train, store


def clean_train(raw):
    """The notebook's fixes to the train table: snake_case, store_id, holiday labels, sort, zero-sales days."""
    df = raw.rename(columns=to_snake_case).rename(columns={"store": "store_id"})
    df = recode_categories(df, "state_holiday", HOLIDAY_LABELS)
    df = sort_rows(df, ["store_id", "date"])
    cleaned = drop_rows_where(df, is_zero_sales_open_day)
    logger.info(f"train: dropped {len(df) - len(cleaned)} open days with zero sales")
    return cleaned


def clean_store(raw):
    """The notebook's fixes to the store table, in the same order."""
    df = raw.rename(columns=to_snake_case).rename(columns={"store": "store_id"})
    df = fix_spelling(df, "promo_interval", PROMO_INTERVAL_FIXES)
    df = recode_categories(df, "assortment", ASSORTMENT_LABELS)
    df = set_missing(df, ["competition_open_since_year", "competition_open_since_month"],
                     df["competition_open_since_year"] < MIN_PLAUSIBLE_YEAR)
    df = fill_missing(df, "competition_distance", df["competition_distance"].median())
    df = combine_year_month(df, "competition_open_since", "competition_open_since_year", "competition_open_since_month")
    df = fill_missing(df, "promo_interval", "none")
    return combine_year_week(df, "promo2_since", "promo2_since_year", "promo2_since_week")


def clean(train_raw, store_raw):
    """Clean both raw tables, check them, and merge them into fct_store_day. Raises if any check fails."""
    train = clean_train(train_raw)
    store = clean_store(store_raw)
    run_checks(train_checks(train, train_raw, HOLIDAY_LABELS) + store_checks(store, store_raw))
    merged = merge_checked(train, store, on="store_id")
    run_checks(merged_checks(merged, train, store, on="store_id"))
    logger.info(f"fct_store_day: {merged.shape[0]:,} rows x {merged.shape[1]} columns")
    return merged
