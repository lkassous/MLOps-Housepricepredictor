"""
Script to register the best model in MLflow Model Registry
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def get_best_run():
    """
    Find the best run based on test_r2 metric
    """
    client = MlflowClient()
    
    # Get the experiment
    experiment = client.get_experiment_by_name("House-Prices-Regression")
    
    if experiment is None:
        print("❌ Experiment 'House-Prices-Regression' not found!")
        print("Please run train_models.py first.")
        return None
    
    # Search for all runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_r2 DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        print("❌ No runs found in the experiment!")
        return None
    
    best_run = runs[0]
    return best_run

def register_best_model():
    """
    Register the best model to MLflow Model Registry
    """
    print("=" * 70)
    print("REGISTERING BEST MODEL TO MLFLOW MODEL REGISTRY")
    print("=" * 70)
    
    # Get the best run
    best_run = get_best_run()
    
    if best_run is None:
        return
    
    # Extract information
    run_id = best_run.info.run_id
    run_name = best_run.data.tags.get('mlflow.runName', 'Unknown')
    test_r2 = best_run.data.metrics.get('test_r2', 0)
    test_rmse = best_run.data.metrics.get('test_rmse', 0)
    test_mae = best_run.data.metrics.get('test_mae', 0)
    
    print(f"\n🏆 Best Model Found:")
    print(f"   Run ID:    {run_id}")
    print(f"   Model:     {run_name}")
    print(f"   Test R²:   {test_r2:.4f}")
    print(f"   Test RMSE: ${test_rmse:,.2f}")
    print(f"   Test MAE:  ${test_mae:,.2f}")
    
    # Model URI
    model_uri = f"runs:/{run_id}/model"
    model_name = "HousePricesPredictor"
    
    print(f"\n📦 Registering model as '{model_name}'...")
    
    # Register the model
    try:
        result = mlflow.register_model(model_uri, model_name)
        print(f"✅ Model registered successfully!")
        print(f"   Model Name: {model_name}")
        print(f"   Version:    {result.version}")
        
        # Transition to Production
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        
        print(f"\n🚀 Model transitioned to 'Production' stage!")
        
        # Add description
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=f"Best performing model: {run_name} with R²={test_r2:.4f}"
        )
        
        print("\n" + "=" * 70)
        print("MODEL REGISTRATION COMPLETE")
        print("=" * 70)
        print(f"\n📊 View in MLflow UI: http://localhost:5000/#/models/{model_name}")
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")

def load_production_model():
    """
    Example: Load the production model for inference
    """
    model_name = "HousePricesPredictor"
    stage = "Production"
    
    print(f"\n📥 Loading model '{model_name}' from stage '{stage}'...")
    
    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

if __name__ == "__main__":
    # Register the best model
    register_best_model()
    
    # Example: Load the production model
    # model = load_production_model()
    # if model:
    #     print("Model ready for predictions!")
