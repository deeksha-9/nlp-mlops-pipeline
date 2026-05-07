# api/main.py

"""
Sentiment Analysis API
Loads model from MLflow and serves predictions using FastAPI
"""

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import logging

# ---------------- LOGGING ---------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FASTAPI APP ---------------- #

app = FastAPI(
    title="Sentiment Analysis API",
    description="Movie Review Sentiment Prediction API",
    version="1.0.0"
)

# ---------------- REQUEST/RESPONSE MODELS ---------------- #

class ReviewRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Movie review text"
    )

class PredictionResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "negative"]
    confidence: float
    model_version: str

# ---------------- MODEL CONFIG ---------------- #

MODEL_NAME = "sentiment-model"
MODEL_VERSION = "2"

model = None

# ---------------- LOAD MODEL ---------------- #

@app.on_event("startup")
async def load_model():
    global model

    try:
        # IMPORTANT: SAME DB USED DURING TRAINING
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

        logger.info(f"Loading model from: {model_uri}")

        # Load model from MLflow registry
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✓ Model loaded successfully")

    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        raise

# ---------------- ROOT ---------------- #

@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API Running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }

# ---------------- HEALTH CHECK ---------------- #

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    }

# ---------------- SINGLE PREDICTION ---------------- #

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        text = request.text.strip()

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        sentiment = "positive" if prediction == 1 else "negative"

        confidence = float(probabilities[prediction])

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION
        }

    except Exception as e:
        logger.error(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ---------------- BATCH PREDICTION ---------------- #

@app.post("/predict/batch")
async def predict_batch(texts: list[str]):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)

        results = []

        for text, pred, probs in zip(texts, predictions, probabilities):

            sentiment = "positive" if pred == 1 else "negative"

            confidence = float(probs[pred])

            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": round(confidence, 4)
            })

        return {
            "count": len(results),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
#uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
#http://127.0.0.1:8000/docs