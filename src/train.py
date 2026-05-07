# src/train.py
"""
Train a sentiment analysis model on IMDB reviews.
Logs all experiments to MLflow for tracking and comparison.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ============================================================
# STEP 1: Load and prepare data
# ============================================================

print("Loading dataset...")
df = pd.read_csv(r"data\IMDB_Dataset.csv")

# Convert sentiment labels to binary (0=negative, 1=positive)
X = df['review']  # Text reviews
y = (df['sentiment'] == 'positive').astype(int)  # 1 for positive, 0 for negative

print(f"Dataset shape: {df.shape}")
print(f"Positive samples: {y.sum()}, Negative samples: {len(y) - y.sum()}")

# ============================================================
# STEP 2: Split into train and test sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Fixed seed for reproducibility
    stratify=y          # Keep same positive/negative ratio in both sets
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ============================================================
# STEP 3: Set up MLflow experiment
# ============================================================

# Create or select an experiment (like a project folder for runs)
mlflow.set_experiment("sentiment-analysis-imdb")

# ============================================================
# STEP 4: Train model and log everything to MLflow
# ============================================================

# Hyperparameters to try (you can change these and retrain)
MAX_FEATURES = 10000  # Keep top 10k most common words
C_VALUE = 1.0         # Regularization strength (lower = more regularization)
MAX_ITER = 500        # Maximum iterations for training

print("\nStarting training...")
print(f"Hyperparameters: max_features={MAX_FEATURES}, C={C_VALUE}")

# Start an MLflow run (this logs everything inside this block)
with mlflow.start_run(run_name="baseline-tfidf-logreg"):
    
    # ---- Log hyperparameters ----
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("C", C_VALUE)
    mlflow.log_param("max_iter", MAX_ITER)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("classifier", "LogisticRegression")
    
    # ---- Build the model pipeline ----
    # Pipeline chains: TF-IDF vectorization → Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=(1, 2),      # Use unigrams and bigrams
            min_df=5,                # Ignore words appearing in < 5 documents
            strip_accents='unicode', # Remove accents
            lowercase=True           # Convert to lowercase
        )),
        ('classifier', LogisticRegression(
            C=C_VALUE,
            max_iter=MAX_ITER,
            random_state=42,
            solver='lbfgs',          # Optimization algorithm
            n_jobs=-1                # Use all CPU cores
        ))
    ])
    
    # ---- Train the model ----
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # ---- Make predictions ----
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # ---- Calculate metrics ----
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    
    # ---- Log metrics to MLflow ----
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    
    # ---- Log the trained model ----
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="sentiment-model"  # Registers in Model Registry
    )
    
    # ---- Print results ----
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print("="*60)
    print("\nModel saved to MLflow!")
    print(f"View results at: http://localhost:5000")

print("\nDone! Run 'mlflow ui' to view the dashboard.")