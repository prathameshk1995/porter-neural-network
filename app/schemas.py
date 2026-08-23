from datetime import datetime
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):

    market_id: float

    created_at: datetime

    store_primary_category: int

    order_protocol: float

    total_items: int = Field(ge=0)

    subtotal: int = Field(ge=0)

    num_distinct_items: int = Field(ge=0)

    min_item_price: int = Field(ge=0)

    max_item_price: int = Field(ge=0)

    total_onshift_dashers: float = Field(ge=0)

    total_busy_dashers: float = Field(ge=0)

    total_outstanding_orders: float = Field(ge=0)

    estimated_store_to_consumer_driving_duration: float = Field(ge=0)


class PredictionResponse(BaseModel):

    predicted_delivery_time_minutes: float