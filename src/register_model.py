# src/register_model.py
"""
Register the latest trained model in MLflow Model Registry
and promote it to Production stage.
"""

import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI to SQLite backend (more reliable than filesystem)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Initialize client
client = MlflowClient()

# Get the experiment - find the latest sentiment-analysis experiment
experiment_name = "sentiment-analysis-imdb"
try:
    experiment = client.get_experiment_by_name(experiment_name)
except:
    experiment = None

if experiment is None:
    print(f"ERROR: Experiment '{experiment_name}' not found!")
    print("Run 'python src/train.py' first to train a model.")
    exit(1)

# Get all runs from this experiment, sorted by accuracy
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_accuracy DESC"],
    max_results=1
)

if not runs:
    print("ERROR: No runs found in this experiment!")
    print("Run 'python src/train.py' first to train a model.")
    exit(1)

# Get the best run
best_run = runs[0]
run_id = best_run.info.run_id
accuracy = best_run.data.metrics.get("test_accuracy", 0)

print(f"Found best run: {run_id}")
print(f"Test Accuracy: {accuracy:.4f}")

# Register the model
model_name = "sentiment-model"
model_uri = f"runs:/{run_id}/model"

try:
    # Try to register (creates new version if model exists)
    model_version = mlflow.register_model(model_uri, model_name)
    version_number = model_version.version
    print(f"✓ Registered model '{model_name}' version {version_number}")
    
except Exception as e:
    print(f"Model registration failed: {e}")
    exit(1)

# Promote to Production stage
try:
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Production",
        archive_existing_versions=True  # Move old Production versions to Archived
    )
    print(f"✓ Promoted version {version_number} to Production stage")
    
except Exception as e:
    print(f"Failed to promote to Production: {e}")
    exit(1)

print("\n" + "="*60)
print("SUCCESS!")
print("="*60)
print(f"Model: {model_name}")
print(f"Version: {version_number}")
print(f"Stage: Production")
print(f"Accuracy: {accuracy:.4f}")
print("="*60)
print("\nYou can now start the API:")
print("uvicorn api.main:app --reload")