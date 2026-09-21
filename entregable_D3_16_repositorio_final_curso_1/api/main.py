from io import StringIO
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from api.dependencies import (
    get_model,
    get_prediction_threshold,
)
from api.schemas import (
    CustomerInput,
    PredictionResponse,
)
from src.models.prediction import predict_customers

app = FastAPI(
    title="NovaTel Customer Churn API",
    description=(
        "API REST para predicción de churn "
        "de clientes de NovaTel."
    ),
    version="1.0.0",
)

PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total number of churn predictions",
    ["prediction"],
)

PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Latency of individual churn predictions",
)

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of predicted churn probability",
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)

INPUT_MONTHLY_FEE = Histogram(
    "churn_input_monthly_fee",
    "Distribution of monthly fee received by the model",
    buckets=(40, 60, 80, 100, 120, 150, 200),
)

INPUT_SUPPORT_CALLS = Histogram(
    "churn_input_support_calls",
    "Distribution of support calls received by the model",
    buckets=(0, 1, 2, 3, 5, 8, 12),
)

INPUT_PAYMENT_DELAY = Histogram(
    "churn_input_payment_delay_days",
    "Distribution of last payment delay received by the model",
    buckets=(0, 3, 7, 15, 30, 60, 90),
)

INPUT_DIGITAL_USAGE = Histogram(
    "churn_input_digital_usage_score",
    "Distribution of digital usage score received by the model",
    buckets=(1, 2, 4, 6, 8, 10),
)

LAST_CHURN_PROBABILITY = Gauge(
    "churn_last_probability",
    "Churn probability from the most recent prediction",
)


@app.get("/")
def root():
    return {
        "message": "NovaTel Customer Churn API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "customer-churn-api",
    }

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerInput):
    try:
        start_time = perf_counter()

        model = get_model()
        threshold = get_prediction_threshold()

        dataframe = pd.DataFrame(
            [customer.model_dump()]
        )

        result = predict_customers(
            model=model,
            dataframe=dataframe,
            threshold=threshold,
        )

        row = result.iloc[0]

        prediction = int(
            row["churn_prediction"]
        )

        PREDICTIONS_TOTAL.labels(
            prediction=str(prediction)
        ).inc()

        probability = float(
            row["churn_probability"]
        )

        CHURN_PROBABILITY.observe(probability)
        LAST_CHURN_PROBABILITY.set(probability)

        INPUT_MONTHLY_FEE.observe(
            customer.monthly_fee
        )
        INPUT_SUPPORT_CALLS.observe(
            customer.support_calls
        )
        INPUT_PAYMENT_DELAY.observe(
            customer.last_payment_delay
        )
        INPUT_DIGITAL_USAGE.observe(
            customer.digital_usage_score
        )

        PREDICTION_LATENCY.observe(
            perf_counter() - start_time
        )

        return PredictionResponse(
            customer_id=(
                str(row["customer_id"])
                if "customer_id" in result.columns
                else None
            ),
            churn_probability=probability,
            churn_prediction=prediction,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A CSV file is required.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty.",
            )

        dataframe = pd.read_csv(
            StringIO(
                contents.decode("utf-8")
            )
        )

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV contains no rows.",
            )

        model = get_model()
        threshold = get_prediction_threshold()

        result = predict_customers(
            model=model,
            dataframe=dataframe,
            threshold=threshold,
        )

        csv_buffer = StringIO()
        result.to_csv(
            csv_buffer,
            index=False,
        )
        csv_buffer.seek(0)

        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; '
                    'filename="customer_predictions.csv"'
                )
            },
        )

    except HTTPException:
        raise

    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="The CSV file must be UTF-8 encoded.",
        ) from exc

    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV could not be parsed.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc