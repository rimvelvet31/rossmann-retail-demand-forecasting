"""Parity check: the API's per-store feature path must build exactly the training features.

For a few stores, it pretends the data ended 7 days earlier, builds those last 7 days the way the API does
(from the store's history only), and compares them with the same rows of the full training table.

Usage, from the repo root: uv run python -m scripts.check_parity
"""

import logging

import pandas as pd

from src.clean import clean, load_raw
from src.config import CATEGORY_COLUMNS, HISTORY_DAYS, HORIZON, NATIONAL_SHARE, STORE_CSV, TRAIN_CSV
from src.features import build_features, category_values, feature_list, find_national_holidays
from src.forecast import PLAN_COLUMNS, build_forecast_rows

logger = logging.getLogger("parity")

SAMPLE_STORES = [1, 2, 13, 85]  # the first store of each type, as in the notebook's forecast table


def main():
    fct = clean(*load_raw(TRAIN_CSV, STORE_CSV))
    national_holidays = find_national_holidays(fct, NATIONAL_SHARE)
    categories = category_values(fct, CATEGORY_COLUMNS)
    features = feature_list(categories)
    full = build_features(fct, national_holidays, categories)

    cutoff = fct["date"].max() - pd.Timedelta(days=HORIZON)  # pretend the data ends here
    for store_id in SAMPLE_STORES:
        store = fct[fct["store_id"] == store_id]
        history = store[(store["date"] > cutoff - pd.Timedelta(days=HISTORY_DAYS)) & (store["date"] <= cutoff)]
        plan = store.loc[store["date"] > cutoff, PLAN_COLUMNS]
        served = build_forecast_rows(history, plan, national_holidays, categories)
        trained = full[(full["store_id"] == store_id) & (full["date"] > cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(served[features], trained[features], check_dtype=False)
        logger.info(f"PASS  store {store_id}: {len(served)} days, all {len(features)} features identical")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
