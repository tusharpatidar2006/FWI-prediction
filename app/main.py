"""
Algerian Forest Fire FWI Prediction API
----------------------------------------
A production-ready FastAPI backend for predicting the Fire Weather Index (FWI)
using a pre-trained Ridge Regression model and a StandardScaler.

Endpoints:
    GET  /          → Root health check
    GET  /health    → Detailed model/scaler status
    POST /predict   → Run FWI prediction
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schema import HealthResponse, PredictionInput, PredictionOutput

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fwi_api")

# ──────────────────────────────────────────────
# Model paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "ridge.pkl"
SCALER_PATH = MODEL_DIR / "scalar.pkl"

# ──────────────────────────────────────────────
# Global model state (loaded at startup)
# ──────────────────────────────────────────────
ml_model = {"ridge": None, "scaler": None}


# ──────────────────────────────────────────────
# Lifespan: load model & scaler once at startup
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts on startup; release on shutdown."""
    logger.info("Starting up — loading ML artifacts...")

    # Load Ridge model
    if not MODEL_PATH.exists():
        logger.critical(f"Model file not found at: {MODEL_PATH}")
        raise FileNotFoundError(f"ridge.pkl not found at {MODEL_PATH}")
    ml_model["ridge"] = joblib.load(MODEL_PATH)
    logger.info(f"Ridge model loaded successfully from {MODEL_PATH}")

    # Load Scaler
    if not SCALER_PATH.exists():
        logger.critical(f"Scaler file not found at: {SCALER_PATH}")
        raise FileNotFoundError(f"scalar.pkl not found at {SCALER_PATH}")
    ml_model["scaler"] = joblib.load(SCALER_PATH)
    logger.info(f"StandardScaler loaded successfully from {SCALER_PATH}")

    logger.info("All ML artifacts loaded. API is ready.")
    yield

    # Shutdown cleanup
    ml_model["ridge"] = None
    ml_model["scaler"] = None
    logger.info("Shutdown complete — ML artifacts released.")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Algerian Forest Fire FWI Prediction API",
    description=(
        "Predicts the **Fire Weather Index (FWI)** using a Ridge Regression model "
        "trained on the Algerian Forest Fires dataset. "
        "Input 9 weather features and receive a continuous FWI score."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ──────────────────────────────────────────────
# CORS middleware (adjust origins for production)
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Global exception handler
# ──────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "status": "error"},
    )


# ──────────────────────────────────────────────
# GET /  →  Root health check
# ──────────────────────────────────────────────
@app.get(
    "/",
    summary="Root health check",
    tags=["Health"],
)
async def root():
    """
    Lightweight ping endpoint. Returns a simple alive message.
    Use GET /health for full model status.
    """
    logger.info("GET / called")
    return {"message": "Algerian Forest Fire FWI Prediction API is running.", "status": "ok"}


# ──────────────────────────────────────────────
# GET /health  →  Detailed model status
# ──────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Detailed health — model & scaler status",
    tags=["Health"],
)
async def health_check():
    """
    Confirms whether the Ridge model and StandardScaler are loaded and ready.
    Returns `model_loaded` and `scaler_loaded` boolean flags.
    """
    logger.info("GET /health called")

    model_loaded = ml_model["ridge"] is not None
    scaler_loaded = ml_model["scaler"] is not None
    all_ready = model_loaded and scaler_loaded

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        model_loaded=model_loaded,
        scaler_loaded=scaler_loaded,
        message="All systems operational." if all_ready else "One or more ML artifacts failed to load.",
    )


# ──────────────────────────────────────────────
# POST /predict  →  FWI prediction
# ──────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict Fire Weather Index (FWI)",
    tags=["Prediction"],
)
async def predict(payload: PredictionInput):
    """
    Accepts 9 weather input features and returns a predicted FWI score.

    **Pipeline:**
    1. Parse & validate input via Pydantic
    2. Assemble features into a NumPy array (exact model order)
    3. Apply `StandardScaler.transform()`
    4. Apply `Ridge.predict()`
    5. Return `{ "prediction": float, "status": "success" }`
    """
    logger.info(f"POST /predict called with input: {payload.model_dump()}")

    # ── Guard: ensure models are available ──
    if ml_model["ridge"] is None or ml_model["scaler"] is None:
        logger.error("Prediction attempted but ML artifacts are not loaded.")
        raise HTTPException(
            status_code=503,
            detail="ML model or scaler is not loaded. Please try again shortly.",
        )

    try:
        # ── Step 1: Build feature array in training order ──
        features = np.array([[
            payload.Temperature,
            payload.RH,
            payload.Ws,
            payload.Rain,
            payload.FFMC,
            payload.DMC,
            payload.DC,
            payload.ISI,
            payload.BUI,
        ]])
        logger.debug(f"Feature array shape: {features.shape}")

        # ── Step 2: Scale features ──
        scaled_features = ml_model["scaler"].transform(features)
        logger.debug(f"Scaled features: {scaled_features}")

        # ── Step 3: Predict ──
        raw_prediction = ml_model["ridge"].predict(scaled_features)
        prediction_value = float(round(raw_prediction[0], 4))

        logger.info(f"Prediction successful: FWI = {prediction_value}")

        return PredictionOutput(prediction=prediction_value, status="success")

    except ValueError as ve:
        logger.error(f"ValueError during prediction: {ve}", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=f"Input data error: {str(ve)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        )