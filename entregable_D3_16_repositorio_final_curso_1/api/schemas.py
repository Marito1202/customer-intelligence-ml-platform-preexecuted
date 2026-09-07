from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customer_id: str | None = Field(
        default=None,
        description="Identificador del cliente",
        examples=["NT-API-001"],
    )

    age: float | None = Field(
        default=None,
        ge=18,
        description="Edad del cliente",
    )

    tenure_months: int = Field(
        ge=0,
        description="Antigüedad del cliente en meses",
    )

    monthly_fee: float = Field(
        ge=0,
        description="Cargo mensual",
    )

    total_spent: float | None = Field(
        default=None,
        ge=0,
        description="Gasto acumulado",
    )

    support_calls: int = Field(
        ge=0,
        description="Número de llamadas a soporte",
    )

    complaints: int = Field(
        ge=0,
        description="Número de reclamos",
    )

    last_payment_delay: int = Field(
        ge=0,
        description="Días de retraso del último pago",
    )

    digital_usage_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    marketing_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    preferred_contact_hour: int = Field(
        ge=0,
        le=23,
    )

    gender: str
    region: str | None = None
    customer_segment: str
    contract_type: str
    internet_service: str | None = None
    tv_service: str
    streaming_service: str
    payment_method: str


class PredictionResponse(BaseModel):
    customer_id: str | None = None
    churn_probability: float
    churn_prediction: int