# NLP Sentiment Analysis MLOps Pipeline

A complete MLOps pipeline for sentiment analysis on movie reviews, featuring model training, experiment tracking, and REST API serving.

## Features

- **Model**: TF-IDF vectorizer + Logistic Regression classifier
- **Dataset**: IMDB movie reviews (50,000 samples)
- **Experiment Tracking**: MLflow for model versioning and metrics
- **API**: FastAPI with interactive documentation
- **Deployment**: Render.com for production serving
- **Automation**: Git-based deployment with auto-retraining

## Project Structure

```
nlp-mlops-pipeline/
├── api/
│   └── main.py              # FastAPI application
├── src/
│   └── train.py             # Model training script
├── data/
│   └── IMDB_Dataset.csv     # Training dataset
├── mlruns/                  # MLflow experiment artifacts
├── render.yaml              # Render deployment config
├── requirements.txt         # Python dependencies
├── verify_setup.py          # Setup verification script
└── README.md
```

## Quick Start

### 1. Local Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python verify_setup.py
```

### 3. Train Model

```bash
python src/train.py
```

**Expected Results:**
- Train Accuracy: ~91%
- Test Accuracy: ~89%
- F1 Score: ~89%

### 4. Run API Locally

```bash
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive documentation.

## API Endpoints

### Health Check
```
GET /health
```

### Single Prediction
```
POST /predict
Content-Type: application/json

{
  "text": "This movie was fantastic!"
}
```

Response:
```json
{
  "text": "This movie was fantastic!",
  "sentiment": "positive",
  "confidence": 0.92,
  "model_version": "1"
}
```

### Batch Prediction
```
POST /predict/batch
Content-Type: application/json

[
  "Great movie!",
  "Terrible film.",
  "Not bad."
]
```

## Model Details

- **Vectorizer**: TF-IDF (max_features=5000, unigrams)
- **Classifier**: Logistic Regression (C=1.0, solver='saga')
- **Train/Test Split**: 80/20
- **Classes**: positive, negative

## Deployment

### Live Application
https://nlp-sentiment-api-xxxxx.onrender.com

### Deploy to Render

```bash
python src/train.py
git add mlruns/ render.yaml
git commit -m "Update trained model"
git push
```

Render auto-deploys on git push.

## MLflow Tracking

View experiments locally:

```bash
mlflow ui
```

Then visit `http://localhost:5000`

## Troubleshooting

**Model not loading:**
- Run: `python src/train.py`
- Check: `ls mlruns/*/models/*/artifacts/`

**Dataset not found:**
- Verify: `data/IMDB_Dataset.csv` exists (63 MB)
- Check: `git ls-files | grep IMDB_Dataset.csv`

**Unicode errors (Windows):**
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## Performance

| Metric | Value |
|--------|-------|
| Train Accuracy | 91.27% |
| Test Accuracy | 89.34% |
| F1 Score | 89.39% |
| Precision | 89.00% |
| Recall | 89.78% |

## Requirements

- Python 3.11+
- 64 MB disk space
- See `requirements.txt` for dependencies

## License

MIT
