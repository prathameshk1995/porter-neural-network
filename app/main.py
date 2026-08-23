from fastapi import FastAPI, HTTPException

from app.predictor import predict
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Porter Delivery ETA Prediction API",
    description=(
        "Neural Network API for predicting"
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

        prediction = predict(input_data)

        return PredictionResponse(
            predicted_delivery_time_minutes=prediction
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )