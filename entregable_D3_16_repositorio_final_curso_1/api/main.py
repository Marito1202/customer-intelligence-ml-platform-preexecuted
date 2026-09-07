from io import StringIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

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


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerInput):
    try:
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

        return PredictionResponse(
            customer_id=(
                str(row["customer_id"])
                if "customer_id" in result.columns
                else None
            ),
            churn_probability=float(
                row["churn_probability"]
            ),
            churn_prediction=int(
                row["churn_prediction"]
            ),
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