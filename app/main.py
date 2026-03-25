import os
import pickle
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schema import FireFeaturesInput, PredictionResponse

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("algerian_forest_api")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

SCALER_PATH = MODEL_DIR / "scalar.pkl"
RIDGE_PATH  = MODEL_DIR / "ridge.pkl"

# ─────────────────────────────────────────────
# Model store  (populated at startup)
# ─────────────────────────────────────────────
ml_models: dict = {}


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# Lifespan  (replaces deprecated on_event)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    logger.info("Loading ML models from %s", MODEL_DIR)
    try:
        ml_models["scaler"] = load_pickle(SCALER_PATH)
        ml_models["ridge"]  = load_pickle(RIDGE_PATH)
        logger.info("Models loaded successfully.")
    except FileNotFoundError as exc:
        logger.error("Startup failed: %s", exc)
        raise RuntimeError(str(exc)) from exc

    yield  # app is running here

    # ── shutdown ──
    ml_models.clear()
    logger.info("Models unloaded. Shutdown complete.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Algerian Forest Fire FWI Predictor",
    description=(
        "Predict the **Fire Weather Index (FWI)** for Algerian forest regions "
        "using a Ridge Regression model trained on the Algerian Forest Fires dataset.\n\n"
        "**Model:** Ridge Regression  \n"
        "**Scaler:** StandardScaler  \n"
        "**Target:** FWI (continuous)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {"message": "Algerian Forest Fire FWI Prediction API is running 🔥"}


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check — confirms models are loaded."""
    models_ok = "scaler" in ml_models and "ridge" in ml_models
    if not models_ok:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    return {"status": "healthy", "models_loaded": list(ml_models.keys())}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: FireFeaturesInput):
    """
    Predict the **Fire Weather Index (FWI)** given weather and fuel-moisture features.

    - Scales input using the pre-fitted StandardScaler.
    - Runs inference with the Ridge Regression model.
    - Returns the predicted FWI value.
    """
    if "scaler" not in ml_models or "ridge" not in ml_models:
        raise HTTPException(status_code=503, detail="Models are not available.")

    try:
        # Build feature vector in the same column order used during training.
        # 9 features after correlation removal (threshold=0.85); 'Classes'
        # column was scaled out — scaler was fitted on weather features only.
        feature_vector = np.array([[
            data.Temperature,
            data.RH,
            data.Ws,
            data.Rain,
            data.FFMC,
            data.DMC,
            data.DC,
            data.ISI,
            data.BUI,
        ]])

        logger.info("Raw input: %s", feature_vector)

        scaled = ml_models["scaler"].transform(feature_vector)
        prediction = ml_models["ridge"].predict(scaled)
        fwi_value = float(round(prediction[0], 4))

        logger.info("Predicted FWI: %s", fwi_value)

        return PredictionResponse(
            fwi_prediction=fwi_value,
            status="Prediction successful",
        )

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}") from exc