# Rossmann Retail Demand Forecasting

Forecast daily Rossmann store sales 7 days ahead, from EDA to a deployed API.

## Problem

Rossmann runs over 3,000 drug stores in Europe. Store managers need to know how much each store will sell in the coming days to plan staffing and stock. A forecast that is too low means empty shelves and lost sales. One that is too high means waste and tied-up cash.

This project trains a regression model on about two and a half years of daily sales, promotions and holidays for 1,115 stores, and forecasts each store's daily sales for the next 7 days.

## How the solution is used

The model is served as a FastAPI web service in Docker, deployed on Google Cloud Run. A planner sends a store, a start date and the planned promotions, and gets 7 daily sales forecasts back.

## Success metric

- **Headline metric:** WAPE (total forecast error divided by total sales), which makes errors comparable across big and small stores.
- **Also reported:** MAE in euros.
- A model only counts as useful if it beats two simple baselines: sales on the same weekday last week, and the store's average for that weekday.

## Dataset

- **Source:** [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data), provided by Dirk Rossmann GmbH through a Kaggle competition.
- **Size:** 1,017,209 rows, one row per store per day, covering 1,115 stores from 2013-01-01 to 2015-07-31.
- **Files used:**
  - `train.csv`: daily sales per store, with customers, promotions and holiday flags.
  - `store.csv`: static store information (store type, assortment, competitor distance, ongoing promotion program).
- **Target:** `Sales` (euros per store per day).
- **Not used:** Kaggle's `test.csv` has no `Sales` column, so the final holdout is taken from the end of `train.csv` instead.

### Data dictionary

| File | Column | Meaning |
|---|---|---|
| train | `Store` | Store ID, 1 to 1,115 |
| train | `Date` | Calendar day |
| train | `DayOfWeek` | 1 = Monday to 7 = Sunday |
| train | `Sales` | Turnover in euros that day (**target**) |
| train | `Customers` | Number of customers that day |
| train | `Open` | 0 = closed, 1 = open |
| train | `Promo` | Whether the store ran a promotion that day |
| train | `StateHoliday` | Public holiday type: `0` none, `a` public, `b` Easter, `c` Christmas |
| train | `SchoolHoliday` | Whether the store was affected by school closures |
| store | `StoreType` | Store model, `a` to `d` |
| store | `Assortment` | Product range level: `a` basic, `b` extra, `c` extended |
| store | `CompetitionDistance` | Distance in metres to the nearest competitor |
| store | `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear` | When the nearest competitor opened |
| store | `Promo2` | Whether the store takes part in a continuing promotion (0 or 1) |
| store | `Promo2SinceWeek`, `Promo2SinceYear` | When the store joined that promotion |
| store | `PromoInterval` | Months when a Promo2 round restarts, e.g. `Feb,May,Aug,Nov` |

### Download

Run this to fetch the data into `data/raw/` (works on Windows, macOS and Linux, no Kaggle account needed):

    uv run python scripts/download_data.py

The original files are on [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and belong to Dirk Rossmann GmbH.

## Approach

- **Horizon:** 7 days ahead. One model with all lag features at least 7 days back (lags 7, 14, 21, 28), reused for each of the 7 forecast days.
- **Validation:** three rolling backtest folds, each trained on earlier data only, to pick and tune models. The last 6 weeks of `train.csv` are kept as a final holdout, scored once. (Kaggle's `test.csv` has no `Sales`, so it isn't used.)
- **Models:** two baselines (same weekday last week, store × weekday average), then Ridge, random forest, and LightGBM/XGBoost.
