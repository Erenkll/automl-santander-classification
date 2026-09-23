from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import SantanderPredictor


app = FastAPI(
    title="Santander Customer Satisfaction API",
    version="1.0.0",
    description="Lightweight inference wrapper for the exported LightGBM model.",
)


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Raw feature dictionary. The predictor reindexes to the stored 100-feature contract.",
    )


_predictor: SantanderPredictor | None = None


def get_predictor() -> SantanderPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SantanderPredictor()
    return _predictor


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        return get_predictor().predict_one(request.features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
