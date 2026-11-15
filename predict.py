"""Forecast API: one store and its 7-day plan in, 7 daily sales forecasts out.

The model bundle is loaded once, when the server starts. 

Endpoints:
- GET  /health   is the service up, and which model it serves
- POST /predict  7 daily sales forecasts (EUR) for one store

Usage, from the repo root: uv run uvicorn predict:app --port 8080
"""

import datetime as dt
import logging
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.config import HORIZON, MODEL_PATH
from src.forecast import PLAN_COLUMNS, forecast, load_bundle


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("predict")

# An example plan for store 1, shown pre-filled on the /docs page (made-up values)
EXAMPLE_REQUEST = {
    "store_id": 1,
    "days": [
        {"date": "2015-08-01", "open": 1, "promo": 0, "state_holiday": "none", "school_holiday": 1},
        {"date": "2015-08-02", "open": 0, "promo": 0, "state_holiday": "none", "school_holiday": 1},
        {"date": "2015-08-03", "open": 1, "promo": 1, "state_holiday": "none", "school_holiday": 1},
        {"date": "2015-08-04", "open": 1, "promo": 1, "state_holiday": "none", "school_holiday": 1},
        {"date": "2015-08-05", "open": 1, "promo": 1, "state_holiday": "none", "school_holiday": 1},
        {"date": "2015-08-06", "open": 1, "promo": 1, "state_holiday": "none", "school_holiday": 1},
        {"date": "2015-08-07", "open": 1, "promo": 1, "state_holiday": "none", "school_holiday": 1},
    ],
}


# ---------- Request and response shapes (Pydantic checks every request against these) ----------

class PlanDay(BaseModel):
    """What the store knows in advance about one day."""
    date: dt.date
    open: Literal[0, 1]
    promo: Literal[0, 1]
    state_holiday: Literal["none", "public", "easter", "christmas"]  # the cleaned labels (HOLIDAY_LABELS)
    school_holiday: Literal[0, 1]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    store_id: int = Field(ge=1)
    days: list[PlanDay] = Field(min_length=HORIZON, max_length=HORIZON)


class DayForecast(BaseModel):
    date: dt.date
    open: int
    sales_forecast: float  # euros, 0 on closed days


class ForecastResponse(BaseModel):
    store_id: int
    model_name: str
    forecasts: list[DayForecast]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    trained_until: dt.date
    holdout_wape: float
    holdout_mae: float


# ---------- App ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the bundle once, when the server starts. A missing file stops the startup with a clear error."""
    app.state.bundle = load_bundle(MODEL_PATH)
    yield


app = FastAPI(
    title="Rossmann sales forecast",
    description="Daily sales forecasts for one Rossmann store, for the 7 days after the data ends.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> HealthResponse:
    """Is the service up, and which model it serves."""
    bundle = request.app.state.bundle
    return HealthResponse(
        status="ok",
        model_name=bundle["model_name"],
        trained_until=bundle["trained_until"].date(),
        holdout_wape=round(bundle["holdout_score"]["wape"], 4),
        holdout_mae=round(bundle["holdout_score"]["mae"], 2),
    )


@app.post("/predict")
def predict(body: ForecastRequest, request: Request) -> ForecastResponse:
    """7 daily sales forecasts (EUR) for one store. Closed days get 0."""
    bundle = request.app.state.bundle
    if body.store_id not in bundle["store_ids"]:
        raise HTTPException(status_code=404, detail=f"unknown store_id {body.store_id}")
    plan = pd.DataFrame([day.model_dump() for day in body.days], columns=PLAN_COLUMNS)
    try:
        result = forecast(bundle, body.store_id, plan)
    except ValueError as error:  # for example, the wrong dates: the caller can fix that
        raise HTTPException(status_code=422, detail=str(error)) from error
    logger.info(f"store {body.store_id}: {result['sales_forecast'].sum():,.0f} EUR over {len(result)} days")
    return ForecastResponse(
        store_id=body.store_id,
        model_name=bundle["model_name"],
        forecasts=[
            DayForecast(date=row.date.date(), open=int(row.open), sales_forecast=float(row.sales_forecast))
            for row in result.itertuples(index=False)
        ],
    )
