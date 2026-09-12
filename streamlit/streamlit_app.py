from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


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
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Porter Delivery ETA Predictor",
    page_icon="🚚",
    layout="centered",
)


# ============================================================
# Load ML Artifacts
# ============================================================

@st.cache_resource
def load_artifacts():

    model = load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    with open(FEATURE_COLUMNS_PATH, "rb") as file:
        feature_columns = pickle.load(file)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return model, scaler, feature_columns, metadata


try:

    model, scaler, feature_columns, model_metadata = (
        load_artifacts()
    )

except Exception as error:

    st.error(
        f"Failed to load ML artifacts: {error}"
    )

    st.stop()


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
    # Convert timestamp
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
    # Remove original timestamp
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
    # Align with training feature columns
    # --------------------------------------------------------

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # Convert to float
    # --------------------------------------------------------

    df = df.astype(float)

    # --------------------------------------------------------
    # Apply the same scaler used during training
    # --------------------------------------------------------

    scaled_data = scaler.transform(df)

    return scaled_data


# ============================================================
# Prediction
# ============================================================

def predict_delivery_time(input_data: dict) -> float:

    processed_data = preprocess_input(input_data)

    prediction = model.predict(
        processed_data,
        verbose=0
    )

    prediction_value = float(
        np.asarray(prediction).flatten()[0]
    )

    return prediction_value


# ============================================================
# Streamlit UI
# ============================================================

st.title("🚚 Porter Delivery ETA Predictor")

st.write(
    "Predict estimated delivery time using a trained "
    "Neural Network model."
)

st.divider()


# ============================================================
# Input Section
# ============================================================

st.subheader("Order Details")


market_id = st.number_input(
    "Market ID",
    min_value=0.0,
    value=1.0,
    step=1.0,
)


created_date = st.date_input(
    "Order Created Date"
)

created_time = st.time_input(
    "Order Created Time"
)

created_at = datetime.combine(
    created_date,
    created_time
)


store_primary_category = st.number_input(
    "Store Primary Category",
    min_value=0,
    value=1,
    step=1,
)


order_protocol = st.number_input(
    "Order Protocol",
    min_value=0.0,
    value=1.0,
    step=1.0,
)


total_items = st.number_input(
    "Total Items",
    min_value=0,
    value=2,
    step=1,
)


subtotal = st.number_input(
    "Subtotal",
    min_value=0,
    value=500,
    step=10,
)


num_distinct_items = st.number_input(
    "Number of Distinct Items",
    min_value=0,
    value=2,
    step=1,
)


min_item_price = st.number_input(
    "Minimum Item Price",
    min_value=0,
    value=100,
    step=10,
)


max_item_price = st.number_input(
    "Maximum Item Price",
    min_value=0,
    value=300,
    step=10,
)


total_onshift_dashers = st.number_input(
    "Total On-Shift Dashers",
    min_value=0.0,
    value=10.0,
    step=1.0,
)


total_busy_dashers = st.number_input(
    "Total Busy Dashers",
    min_value=0.0,
    value=5.0,
    step=1.0,
)


total_outstanding_orders = st.number_input(
    "Total Outstanding Orders",
    min_value=0.0,
    value=5.0,
    step=1.0,
)


estimated_store_to_consumer_driving_duration = st.number_input(
    "Estimated Store-to-Consumer Driving Duration (seconds)",
    min_value=0.0,
    value=1000.0,
    step=100.0,
)


# ============================================================
# Create Input Dictionary
# ============================================================

input_data = {

    "market_id": market_id,

    "created_at": created_at,

    "store_primary_category": store_primary_category,

    "order_protocol": order_protocol,

    "total_items": total_items,

    "subtotal": subtotal,

    "num_distinct_items": num_distinct_items,

    "min_item_price": min_item_price,

    "max_item_price": max_item_price,

    "total_onshift_dashers": total_onshift_dashers,

    "total_busy_dashers": total_busy_dashers,

    "total_outstanding_orders": total_outstanding_orders,

    "estimated_store_to_consumer_driving_duration":
        estimated_store_to_consumer_driving_duration,
}


# ============================================================
# Prediction Button
# ============================================================

st.divider()

if st.button(
    "Predict Delivery Time",
    type="primary",
    use_container_width=True,
):

    try:

        prediction = predict_delivery_time(
            input_data
        )

        st.success(
            "Prediction generated successfully!"
        )

        st.metric(
            label="Predicted Delivery Time",
            value=f"{prediction:.2f} minutes",
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# Model Information
# ============================================================

with st.expander("Model Information"):

    st.write(
        f"**Model Input Features:** "
        f"{len(feature_columns)}"
    )

    st.write(
        f"**Model Input Shape:** "
        f"{model.input_shape}"
    )

    if model_metadata:

        st.write("**Metadata:**")

        st.json(model_metadata)