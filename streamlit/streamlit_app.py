import requests
import streamlit as st
from datetime import datetime


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Porter Delivery ETA Predictor",
    page_icon="🚚",
    layout="centered"
)


# --------------------------------------------------
# API Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8001"


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🚚 Porter Delivery ETA Predictor")

st.write(
    "Predict the estimated delivery time for a Porter order "
    "using a Neural Network model."
)

st.divider()


# --------------------------------------------------
# Input Form
# --------------------------------------------------

with st.form("prediction_form"):

    st.subheader("Order Information")

    market_id = st.number_input(
        "Market ID",
        min_value=1.0,
        value=1.0,
        step=1.0
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
        value=38,
        step=1
    )

    order_protocol = st.number_input(
        "Order Protocol",
        min_value=0.0,
        value=1.0,
        step=1.0
    )

    total_items = st.number_input(
        "Total Items",
        min_value=0,
        value=4,
        step=1
    )

    subtotal = st.number_input(
        "Subtotal",
        min_value=0,
        value=5800,
        step=100
    )

    num_distinct_items = st.number_input(
        "Number of Distinct Items",
        min_value=0,
        value=4,
        step=1
    )

    min_item_price = st.number_input(
        "Minimum Item Price",
        min_value=0,
        value=700,
        step=100
    )

    max_item_price = st.number_input(
        "Maximum Item Price",
        min_value=0,
        value=2000,
        step=100
    )

    total_onshift_dashers = st.number_input(
        "Total Onshift Dashers",
        min_value=0.0,
        value=19.0,
        step=1.0
    )

    total_busy_dashers = st.number_input(
        "Total Busy Dashers",
        min_value=0.0,
        value=19.0,
        step=1.0
    )

    total_outstanding_orders = st.number_input(
        "Total Outstanding Orders",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

    estimated_store_to_consumer_driving_duration = st.number_input(
        "Estimated Store-to-Consumer Driving Duration (seconds)",
        min_value=0.0,
        value=344.0,
        step=10.0
    )

    submitted = st.form_submit_button(
        "Predict Delivery Time"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    payload = {
        "market_id": market_id,
        "created_at": created_at.isoformat(),
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
            estimated_store_to_consumer_driving_duration
    }

    try:

        with st.spinner("Generating prediction..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            prediction = result[
                "predicted_delivery_time_minutes"
            ]

            st.success("Prediction generated successfully!")

            st.metric(
                label="Predicted Delivery Time",
                value=f"{prediction:.2f} minutes"
            )

        else:

            st.error(
                f"Prediction failed: {response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI server. "
            "Please make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )