import numpy as np

from app.predictor import predict


def test_prediction_parity():

    # ---------------------------------------------------------
    # Same raw record used during manual parity testing
    # ---------------------------------------------------------

    input_data = {
        "market_id": 1.0,
        "created_at": "2015-02-17T03:13:20",
        "store_primary_category": 38,
        "order_protocol": 1.0,
        "total_items": 4,
        "subtotal": 5800,
        "num_distinct_items": 4,
        "min_item_price": 700,
        "max_item_price": 2000,
        "total_onshift_dashers": 19.0,
        "total_busy_dashers": 19.0,
        "total_outstanding_orders": 30.0,
        "estimated_store_to_consumer_driving_duration": 344.0,
    }

    # ---------------------------------------------------------
    # Expected prediction obtained from original notebook
    # ---------------------------------------------------------

    expected_prediction =  53.332664489746094                    # PUT YOUR NOTEBOOK VALUE HERE

    # ---------------------------------------------------------
    # Production prediction
    # ---------------------------------------------------------

    actual_prediction = predict(input_data)

    # ---------------------------------------------------------
    # Compare predictions
    # ---------------------------------------------------------

    assert np.isclose(
        expected_prediction,
        actual_prediction,
        rtol=1e-5,
        atol=1e-5,
    ), (
        f"Prediction mismatch: "
        f"expected={expected_prediction}, "
        f"actual={actual_prediction}"
    )