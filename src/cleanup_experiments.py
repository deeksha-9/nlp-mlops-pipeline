"""
Clean up deleted experiments in MLflow.
This script permanently removes deleted experiments so we can start fresh.
"""

import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI to SQLite backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

print("Fetching all experiments (including deleted)...")
try:
    # Get ALL experiments including deleted ones
    experiments = client.search_experiments()
    
    for exp in experiments:
        print(f"Found experiment: {exp.name} (ID: {exp.experiment_id}, State: {exp.lifecycle_stage})")
        
        if "sentiment-analysis-imdb" in exp.name:
            print(f"  → Restoring experiment {exp.name}...")
            try:
                # Restore the deleted experiment
                client.restore_experiment(exp.experiment_id)
                print(f"  ✓ Restored {exp.name}")
            except Exception as e:
                print(f"  Error restoring: {e}")
    
    print("\nDone!")
    
except Exception as e:
    print(f"Error: {e}")
