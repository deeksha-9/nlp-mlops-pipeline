# api/main.py
"""
Sentiment Analysis API
Loads a trained model from MLflow (SQLite backend) and serves predictions via REST API.
"""

import os
import glob
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# STEP 1: Initialize FastAPI app
# ============================================================

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment (positive/negative) of movie reviews using ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================
# STEP 2: Define request and response models
# ============================================================

class ReviewRequest(BaseModel):
    """Input schema for prediction requests"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Movie review text to analyze",
        example="This movie was absolutely fantastic! Great acting and plot."
    )

class PredictionResponse(BaseModel):
    """Output schema for prediction responses"""
    text: str = Field(description="The input text that was analyzed")
    sentiment: Literal["positive", "negative"] = Field(description="Predicted sentiment")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score (0-1)")
    model_version: str = Field(description="Model version used")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str

# ============================================================
# STEP 3: Load model from MLflow on startup
# ============================================================

MODEL_NAME = "sentiment-model"
model = None
model_path = None

@app.on_event("startup")
async def load_model():
    """
    Load the model from MLflow - works with SQLite backend.
    """
    global model, model_path
    
    try:
        # Use SQLite backend (matching your train.py)
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(mlflow_uri)
        
        logger.info("Starting up...")
        logger.info(f"MLflow tracking URI: {mlflow_uri}")
        logger.info("Loading model from MLflow registry...")
        
        # Try loading from Model Registry first
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Get all versions of the model
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            
            if not versions:
                raise Exception(f"No versions found for model '{MODEL_NAME}'")
            
            # Get the latest version number
            latest_version = max([int(v.version) for v in versions])
            
            model_uri = f"models:/{MODEL_NAME}/{latest_version}"
            logger.info(f"Loading model: {model_uri}")
            
            model = mlflow.sklearn.load_model(model_uri)
            model_path = f"version-{latest_version}"
            
            logger.info("✓ Model loaded successfully from registry!")
            logger.info(f"  Model: {MODEL_NAME}")
            logger.info(f"  Version: {latest_version}")
            
        except Exception as e1:
            logger.warning(f"Registry load failed: {e1}")
            logger.info("Trying to load from run artifacts...")
            
            # Fallback: Load from run artifacts
            model_paths = []
            
            # Search in both possible locations
            for pattern in [
                "./mlruns/*/*/artifacts/model",
                "./mlruns/*/models/*/artifacts"
            ]:
                model_paths.extend(glob.glob(pattern))
            
            if not model_paths:
                raise Exception(
                    "No model found! Make sure 'python src/train.py' completed successfully. "
                    f"Original error: {e1}"
                )
            
            # Get the most recently created model
            model_path = max(model_paths, key=os.path.getmtime)
            
            logger.info(f"Loading from artifacts: {model_path}")
            model = mlflow.sklearn.load_model(model_path)
            
            logger.info("✓ Model loaded successfully from artifacts!")
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        logger.error("Make sure:")
        logger.error("  1. python src/train.py completed successfully")
        logger.error("  2. Model was registered in MLflow")
        raise

# ============================================================
# STEP 4: Define API endpoints
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - shows API info"""
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns 200 if API is running and model is loaded.
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: ReviewRequest):
    """
    Predict sentiment of a movie review.
    
    **Input:** Text of a movie review
    
    **Output:** Sentiment (positive/negative) with confidence score
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )
    
    try:
        # Get input text
        text = request.text.strip()
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        # Make prediction
        prediction = model.predict([text])[0]  # Returns 0 or 1
        probabilities = model.predict_proba([text])[0]  # Returns [prob_neg, prob_pos]
        
        # Convert to human-readable format
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(probabilities[prediction])  # Confidence in the predicted class
        
        # Log the prediction
        logger.info(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "model_version": model_path if model_path else "unknown"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(texts: list[str]):
    """
    Predict sentiment for multiple reviews at once.
    
    **Input:** List of review texts (max 100)
    
    **Output:** List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per batch")
    
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
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# STEP 5: Run the app (for development only)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
#uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
#http://127.0.0.1:8000/docs