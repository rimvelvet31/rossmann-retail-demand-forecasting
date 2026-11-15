# Forecast API image: Python 3.12, runtime dependencies only, the model bundle baked in.
# Build after train.py has created models/forecast_bundle.pkl:
#   docker build -t forecast-api:v1 .
#   docker run --rm -p 8080:8080 forecast-api:v1
FROM python:3.12-slim

# uv, pinned to the version that wrote uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

# LightGBM needs the OpenMP library, which the slim image leaves out
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# Dependencies first: this layer is reused until pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# Then the app code and the model bundle
COPY src/ src/
COPY predict.py ./
COPY models/ models/

# Run as a normal user, not root
RUN useradd --create-home app
USER app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
