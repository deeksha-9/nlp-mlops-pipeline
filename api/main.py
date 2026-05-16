# api/main.py
"""
Sentiment Analysis API
Loads a trained model from MLflow and serves predictions via REST API.
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
model_version = None

@app.on_event("startup")
async def load_model():
    """
    Load the model from MLflow using file-based tracking.
    Tries registry first, falls back to run artifacts.
    """
    import os
    global model, model_version
    
    try:
        # Use file-based tracking (same as training)
        mlflow.set_tracking_uri("./mlruns")
        
        logger.info("=" * 60)
        logger.info("STARTUP: Loading model...")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"MLflow tracking URI: ./mlruns")
        logger.info(f"MLflow directory exists: {os.path.exists('./mlruns')}")
        logger.info("=" * 60)
        
        # Strategy 1: Try loading from Model Registry
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Get all versions of the model
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            
            if versions:
                # Get the latest version
                latest = max(versions, key=lambda v: int(v.version))
                model_version = latest.version
                
                model_uri = f"models:/{MODEL_NAME}/{model_version}"
                logger.info(f"Loading from registry: {model_uri}")
                
                model = mlflow.sklearn.load_model(model_uri)
                
                logger.info("✓ Model loaded successfully from registry!")
                logger.info(f"  Model: {MODEL_NAME}")
                logger.info(f"  Version: {model_version}")
                return
            else:
                raise Exception("No model versions found in registry")
                
        except Exception as e1:
            logger.warning(f"Registry load failed: {e1}")
            logger.info("Trying to load from run artifacts...")
        
        # Strategy 2: Load from run artifacts (fallback)
        model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        
        if not model_paths:
            # Show debug info
            logger.error("No model found!")
            logger.error("Directory contents:")
            for root, dirs, files in os.walk("./mlruns"):
                level = root.replace("./mlruns", "").count(os.sep)
                if level < 3:  # Don't go too deep
                    indent = " " * 2 * level
                    logger.error(f"{indent}{os.path.basename(root)}/")
            
            raise Exception(
                "No model found! Ensure 'python src/train.py' ran successfully during build."
            )
        
        # Get the most recently created model
        latest_model_path = max(model_paths, key=os.path.getmtime)
        
        logger.info(f"Loading from artifacts: {latest_model_path}")
        model = mlflow.sklearn.load_model(latest_model_path)
        model_version = latest_model_path.split(os.sep)[2]  # Extract run ID
        
        logger.info("✓ Model loaded successfully from artifacts!")
        logger.info(f"  Path: {latest_model_path}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        logger.error("Troubleshooting:")
        logger.error("  1. Check that 'python src/train.py' completed in build logs")
        logger.error("  2. Verify data/IMDB Dataset.csv exists")
        logger.error("  3. Check build command includes: python src/train.py")
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
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )
    
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        # Make prediction
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(probabilities[prediction])
        
        logger.info(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "model_version": str(model_version) if model_version else "unknown"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )