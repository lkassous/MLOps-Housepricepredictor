"""
MLflow Training Pipeline for House Prices Prediction
Uses MLflow to track experiments and compare multiple regression models
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Import data preparation functions
from data_preparation import load_data, preprocess_data, split_data

def evaluate_model(y_true, y_pred):
    """
    Calculate regression metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Train a model and log results to MLflow
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate on training set
        train_metrics = evaluate_model(y_train, y_pred_train)
        
        # Evaluate on test set
        test_metrics = evaluate_model(y_test, y_pred_test)
        
        # Log model name
        mlflow.log_param('model_name', model_name)
        
        # Log model-specific parameters
        params = model.get_params()
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float, str, bool)) or param_value is None:
                mlflow.log_param(param_name, param_value)
        
        # Log training metrics
        mlflow.log_metric('train_rmse', train_metrics['rmse'])
        mlflow.log_metric('train_mae', train_metrics['mae'])
        mlflow.log_metric('train_r2', train_metrics['r2_score'])
        
        # Log test metrics
        mlflow.log_metric('test_rmse', test_metrics['rmse'])
        mlflow.log_metric('test_mae', test_metrics['mae'])
        mlflow.log_metric('test_r2', test_metrics['r2_score'])
        
        # Log the model
        mlflow.sklearn.log_model(model, 'model')
        
        # Print results
        print(f"\nTraining Results:")
        print(f"  RMSE: ${train_metrics['rmse']:,.2f}")
        print(f"  MAE:  ${train_metrics['mae']:,.2f}")
        print(f"  R²:   {train_metrics['r2_score']:.4f}")
        
        print(f"\nTest Results:")
        print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
        print(f"  MAE:  ${test_metrics['mae']:,.2f}")
        print(f"  R²:   {test_metrics['r2_score']:.4f}")
        
        return test_metrics

def main():
    """
    Main training pipeline
    """
    print("=" * 70)
    print("HOUSE PRICES PREDICTION - MLFLOW EXPERIMENT TRACKING")
    print("=" * 70)
    
    # Set MLflow experiment
    mlflow.set_experiment("House-Prices-Regression")
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    df = load_data('train.csv')
    X, y, label_encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Define models to evaluate
    print("\n2. Defining models to evaluate...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
    }
    
    print(f"Total models to train: {len(models)}")
    
    # Train and log all models
    print("\n3. Training models and logging to MLflow...")
    results = {}
    
    for model_name, model in models.items():
        metrics = train_and_log_model(model_name, model, X_train, X_test, y_train, y_test)
        results[model_name] = metrics
    
    # Display summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY OF ALL MODELS")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Test RMSE':<15} {'Test MAE':<15} {'Test R²':<10}")
    print("-" * 70)
    
    # Sort by R² score (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} ${metrics['rmse']:>13,.2f} ${metrics['mae']:>13,.2f} {metrics['r2_score']:>9.4f}")
    
    # Identify best model
    best_model = sorted_results[0][0]
    best_metrics = sorted_results[0][1]
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"   Test RMSE: ${best_metrics['rmse']:,.2f}")
    print(f"   Test MAE:  ${best_metrics['mae']:,.2f}")
    print(f"   Test R²:   {best_metrics['r2_score']:.4f}")
    print("=" * 70)
    
    print("\n✅ All experiments logged to MLflow!")
    print("📊 View results at: http://localhost:5000")
    print("\nNext steps:")
    print("  1. Compare models in MLflow UI")
    print("  2. Register the best model")
    print("  3. Deploy to production")

if __name__ == "__main__":
    main()
