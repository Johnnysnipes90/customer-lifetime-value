from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import CLVPredictor

predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts once at startup and keep them in memory.
    """
    global predictor
    predictor = CLVPredictor()
    yield
    predictor = None


app = FastAPI(
    title="Customer Lifetime Value Prediction API",
    description=(
        "A production-style FastAPI service for predicting customer lifetime value "
        "using a two-stage machine learning architecture: "
        "P(return) × E(value | return)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class CLVRequest(BaseModel):
    total_revenue: float = Field(
        ..., ge=0, description="Total historical customer revenue"
    )
    total_orders: int = Field(..., ge=0, description="Total historical orders")
    total_items: int = Field(
        ..., ge=0, description="Total historical quantity purchased"
    )
    avg_line_revenue: float = Field(
        ..., ge=0, description="Average revenue per transaction line"
    )
    avg_unit_price: float = Field(..., ge=0, description="Average price per item")
    unique_products: int = Field(
        ..., ge=0, description="Number of unique products purchased"
    )
    unique_countries: int = Field(
        ..., ge=0, description="Number of unique countries purchased from"
    )
    customer_tenure_days: int = Field(..., ge=0, description="Customer tenure in days")
    recency_days: int = Field(..., ge=0, description="Days since last purchase")
    avg_revenue_per_order: float = Field(
        ..., ge=0, description="Average revenue per order"
    )
    active_days: int = Field(..., ge=0, description="Number of distinct purchase days")
    avg_days_between_orders: float = Field(
        ..., ge=0, description="Average number of days between orders"
    )
    revenue_per_day: float = Field(
        ..., ge=0, description="Revenue intensity over tenure"
    )
    orders_per_day: float = Field(..., ge=0, description="Order intensity over tenure")
    recency_ratio: float = Field(..., ge=0, description="Recency normalized by tenure")
    items_per_order: float = Field(
        ..., ge=0, description="Average items purchased per order"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_revenue": 1603.68,
                "total_orders": 3,
                "total_items": 956,
                "avg_line_revenue": 25.0575,
                "avg_unit_price": 2.877344,
                "unique_products": 50,
                "unique_countries": 1,
                "customer_tenure_days": 116,
                "recency_days": 39,
                "avg_revenue_per_order": 534.56,
                "active_days": 3,
                "avg_days_between_orders": 58.0,
                "revenue_per_day": 13.82,
                "orders_per_day": 0.0259,
                "recency_ratio": 0.3362,
                "items_per_order": 318.67,
            }
        }
    }


class CLVResponse(BaseModel):
    return_probability: float
    predicted_value_if_return: float
    expected_clv: float


class SHAPFactor(BaseModel):
    feature: str
    feature_value: float | int | str
    shap_value: float
    direction: str


class CLVExplanationResponse(BaseModel):
    top_factors: List[SHAPFactor]
    all_factors: List[SHAPFactor]


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    return {
        "message": "Customer Lifetime Value Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "explain": "/explain",
    }


@app.get("/health", tags=["General"])
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "service": "clv-api",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=CLVResponse, tags=["Prediction"])
def predict_clv(request: CLVRequest) -> CLVResponse:
    """
    Predict customer lifetime value using the two-stage CLV system.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_data = request.model_dump()
        prediction = predictor.predict(input_data)
        return CLVResponse(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain", response_model=CLVExplanationResponse, tags=["Prediction"])
def explain_clv(request: CLVRequest) -> CLVExplanationResponse:
    """
    Explain the conditional CLV prediction using SHAP.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_data = request.model_dump()
        explanation = predictor.explain_prediction(input_data)
        return CLVExplanationResponse(**explanation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")