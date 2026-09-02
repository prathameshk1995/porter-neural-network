from pathlib import Path
import json
import pickle
import logging

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger(__name__)


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
# Configuration
# ============================================================

CATEGORICAL_COLUMNS = [
    "market_id",
    "store_primary_category",
    "order_protocol",
]

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
# Artifact loading
# ============================================================

def load_artifacts():

    logger.info("Loading ML artifacts...")

    # Check files exist
    artifact_paths = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURE_COLUMNS_PATH,
        METADATA_PATH,
    ]

    for path in artifact_paths:

        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    # Load model
    loaded_model = load_model(MODEL_PATH)

    # Load scaler
    with open(SCALER_PATH, "rb") as file:
        loaded_scaler = pickle.load(file)

    # Load feature columns
    with open(FEATURE_COLUMNS_PATH, "rb") as file:
        loaded_feature_columns = pickle.load(file)

    # Load metadata
    with open(METADATA_PATH, "r") as file:
        loaded_metadata = json.load(file)

    logger.info("ML artifacts loaded successfully.")

    return (
        loaded_model,
        loaded_scaler,
        loaded_feature_columns,
        loaded_metadata,
    )


# ============================================================
# Load artifacts
# ============================================================

model, scaler, feature_columns, model_metadata = load_artifacts()


# ============================================================
# Artifact validation
# ============================================================

def validate_artifacts():

    logger.info("Validating ML artifacts...")

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    if not isinstance(feature_columns, list):
        raise ValueError(
            "feature_columns.pkl must contain a list."
        )

    if len(feature_columns) != 75:
        raise ValueError(
            f"Expected 75 feature columns, "
            f"found {len(feature_columns)}."
        )

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None
    )

    if scaler_features != len(feature_columns):
        raise ValueError(
            "Scaler and feature columns mismatch: "
            f"scaler={scaler_features}, "
            f"features={len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_input_features = model.input_shape[-1]

    if model_input_features != len(feature_columns):
        raise ValueError(
            "Model and feature columns mismatch: "
            f"model={model_input_features}, "
            f"features={len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    logger.info(
        "Model features: %s",
        len(feature_columns)
    )

    logger.info(
        "Scaler features: %s",
        scaler_features
    )

    logger.info(
        "Model input features: %s",
        model_input_features
    )

    logger.info("Artifact validation successful.")


# Validate immediately when application starts
validate_artifacts()


# ============================================================
# Preprocessing
# ============================================================

def preprocess_input(input_data: dict) -> np.ndarray:

    df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    # --------------------------------------------------------
    # Datetime conversion
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
    # One-hot encoding
    # --------------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int
    )

    # --------------------------------------------------------
    # Align with training features
    # --------------------------------------------------------

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df = df.astype(float)

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaled_data = scaler.transform(df)

    return scaled_data


# ============================================================
# Prediction
# ============================================================

def predict(input_data: dict) -> float:

    processed_data = preprocess_input(input_data)

    prediction = model.predict(
        processed_data,
        verbose=0
    )

    prediction_value = float(
        np.asarray(prediction).flatten()[0]
    )

    return prediction_value