from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


# ============================================================
# Load artifacts ONCE when application starts
# ============================================================

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

with open(FEATURE_COLUMNS_PATH, "rb") as file:
    feature_columns = pickle.load(file)

with open(METADATA_PATH, "r") as file:
    model_metadata = json.load(file)


# ============================================================
# Configuration
# ============================================================

CATEGORICAL_COLUMNS = [
    "market_id",
    "store_primary_category",
    "order_protocol",
]


# ============================================================
# Input validation
# ============================================================

REQUIRED_COLUMNS = [
    "market_id",
    "created_at",
    "store_primary_category",
    "order_protocol",
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    "estimated_store_to_consumer_driving_duration",
]


# ============================================================
# Preprocessing
# ============================================================

def preprocess_input(input_data: dict) -> np.ndarray:
    """
    Apply the exact preprocessing steps used during training.

    Steps:
        1. Convert input dictionary to DataFrame
        2. Convert created_at to datetime
        3. Extract hour_of_day
        4. Extract day_of_week
        5. Remove created_at
        6. One-hot encode categorical columns
        7. Align columns to the 75 model features
        8. Apply StandardScaler
    """

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    # --------------------------------------------------------
    # Convert created_at to datetime
    # --------------------------------------------------------

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="raise"
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df["hour_of_day"] = df["created_at"].dt.hour

    df["day_of_week"] = df["created_at"].dt.dayofweek

    # --------------------------------------------------------
    # Remove datetime column
    # --------------------------------------------------------

    df.drop(
        columns=["created_at"],
        inplace=True
    )

    # --------------------------------------------------------
    # One-hot encode categorical features
    # --------------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int
    )

    # --------------------------------------------------------
    # Align with EXACT training features
    # --------------------------------------------------------

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # Ensure numeric values
    # --------------------------------------------------------

    df = df.astype(float)

    # --------------------------------------------------------
    # Scale using fitted StandardScaler
    # --------------------------------------------------------

    scaled_data = scaler.transform(df)

    return scaled_data


# ============================================================
# Prediction
# ============================================================

def predict(input_data: dict) -> float:
    """
    Generate delivery-time prediction for one record.
    """

    processed_data = preprocess_input(input_data)

    prediction = model.predict(
        processed_data,
        verbose=0
    )

    prediction_value = float(
        np.asarray(prediction).flatten()[0]
    )

    return prediction_value