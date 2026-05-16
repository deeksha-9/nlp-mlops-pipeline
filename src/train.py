# src/train.py
"""
Train a sentiment analysis model on IMDB reviews.
Logs all experiments to MLflow and registers the model.
Uses file-based tracking (./mlruns) for compatibility with Render.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mlflow.tracking import MlflowClient

# ============================================================
# STEP 1: Load and prepare data
# ============================================================

import os
import sys

print("Loading dataset...")
print(f"Current working directory: {os.getcwd()}")
print(f"Dataset path: data/IMDB_Dataset.csv")
print(f"File exists: {os.path.exists('data/IMDB_Dataset.csv')}")

try:
    df = pd.read_csv("data/IMDB_Dataset.csv")
except FileNotFoundError as e:
    print(f"ERROR: Dataset file not found: {e}")
    print(f"Files in data directory: {os.listdir('data') if os.path.exists('data') else 'data/ does not exist'}")
    sys.exit(1)

# Convert sentiment labels to binary (0=negative, 1=positive)
X = df['review']
y = (df['sentiment'] == 'positive').astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Positive samples: {y.sum()}, Negative samples: {len(y) - y.sum()}")

# ============================================================
# STEP 2: Split into train and test sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ============================================================
# STEP 3: Set up MLflow experiment
# ============================================================

# Use file-based tracking (works on both local and Render)
mlflow.set_tracking_uri("./mlruns")
print(f"MLflow tracking URI: ./mlruns")

mlflow.set_experiment("sentiment-analysis-imdb")

# ============================================================
# STEP 4: Train model and log everything to MLflow
# ============================================================

# Hyperparameters - OPTIMIZED FOR DEPLOYMENT (fast training)
MAX_FEATURES = 5000  # Reduced from 10000 for faster vectorization
C_VALUE = 1.0
MAX_ITER = 100  # Reduced from 500 for faster convergence

print("\nStarting training...")
print(f"Hyperparameters: max_features={MAX_FEATURES}, C={C_VALUE}, solver=saga")

with mlflow.start_run(run_name="baseline-tfidf-logreg"):
    
    # Log hyperparameters
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("C", C_VALUE)
    mlflow.log_param("max_iter", MAX_ITER)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("classifier", "LogisticRegression")
    mlflow.log_param("solver", "saga")  # Faster than lbfgs for large datasets
    
    # Build the model pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=(1, 1),  # Changed from (1,2) - unigrams only for speed
            min_df=5,
            strip_accents='unicode',
            lowercase=True,
            dtype='float32'  # Use float32 instead of float64 for speed
        )),
        ('classifier', LogisticRegression(
            C=C_VALUE,
            max_iter=MAX_ITER,
            random_state=42,
            solver='saga',  # Changed from lbfgs - faster for large datasets
            n_jobs=-1,
            tol=0.01  # Increased tolerance for faster convergence
        ))
    ])
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    
    # Log metrics to MLflow
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    
    # Log and register the trained model
    model_info = mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="sentiment-model"
    )
    
    # Get the model version
    client = MlflowClient()
    model_version = model_info.registered_model_version
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print("="*60)
    print(f"\n[OK] Model saved to MLflow!")
    print(f"[OK] Model registered as 'sentiment-model' version {model_version}")
    print(f"Model location: ./mlruns")

print("\n[OK] Done! Model is ready for deployment.")