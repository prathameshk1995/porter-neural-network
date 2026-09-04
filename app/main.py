import logging

from fastapi import FastAPI, HTTPException

from app.predictor import predict
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)


# ============================================================
# Logging configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Porter Delivery ETA Prediction API",
    description=(
        "Neural Network API for predicting "
        "Porter delivery time in minutes."
    ),
    version="1.0.0",
)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Porter Delivery ETA Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_delivery_time(
    request: PredictionRequest
):

    try:

        input_data = request.model_dump()

        logger.info(
            "Prediction request received."
        )

        prediction = predict(input_data)

        logger.info(
            "Prediction generated successfully: %.4f minutes",
            prediction
        )

        return PredictionResponse(
            predicted_delivery_time_minutes=prediction
        )

    except ValueError as error:

        logger.error(
            "Invalid prediction input: %s",
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        logger.exception(
            "Unexpected prediction error."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error."
        )