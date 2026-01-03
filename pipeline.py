"""
MLOps Pipeline for House Price Prediction
==========================================
Automated pipeline that:
1. Loads and validates data from CSV
2. Cleans and preprocesses data
3. Trains 3 models (Linear Regression, Random Forest, XGBoost)
4. Evaluates and compares models
5. Auto-selects best model
6. Registers/updates model in MLflow production
7. Generates reports

Usage:
    python pipeline.py --data-path train.csv --output-path ./mlruns
    
Author: MLOps Team
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MLFLOW_EXPERIMENT_NAME = "House-Prices-Production-Pipeline"
MODEL_REGISTRY_NAME = "HousePricesPredictor"
TARGET_COLUMN = "SalePrice"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Categorical columns (43 in this dataset)
CATEGORICAL_COLS = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
    'SaleCondition'
]

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    logger.info(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Validate data quality and return report."""
    logger.info("Validating data quality...")
    
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_stats': df.describe().to_dict()
    }
    
    # Check for target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    logger.info(f"✅ Data validation complete")
    logger.info(f"   Missing values: {report['duplicates']} duplicates")
    
    return report


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess data:
    1. Handle missing values
    2. Encode categorical variables
    3. Separate features and target
    4. Split into train/test
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoders, feature_names
    """
    logger.info("Starting data preprocessing...")
    
    # Make a copy
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"✅ Duplicates removed: {len(df)} rows remaining")
    
    # Separate features and target
    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)
    
    # Handle missing values for numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Handle missing values for categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna('Missing', inplace=True)
    
    logger.info(f"✅ Missing values handled")
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
    
    logger.info(f"✅ Categorical variables encoded: {len(label_encoders)} encoders")
    
    # Store feature names
    feature_names = X_encoded.columns.tolist()
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    logger.info(f"✅ Data split: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"   Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, label_encoders, feature_names


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_and_evaluate_model(model, model_name: str, X_train, X_test, y_train, y_test):
    """Train model and return metrics."""
    logger.info(f"Training {model_name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    logger.info(f"✅ {model_name} trained")
    logger.info(f"   Test RMSE: ${test_rmse:,.2f}")
    logger.info(f"   Test R²: {test_r2:.4f}")
    
    return model, metrics, y_test_pred


def train_models(X_train, X_test, y_train, y_test):
    """Train all 3 models."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING ALL MODELS")
    logger.info("="*70)
    
    models_config = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        )
    }
    
    trained_models = {}
    all_metrics = {}
    predictions = {}
    
    for model_name, model in models_config.items():
        model, metrics, y_pred = train_and_evaluate_model(
            model, model_name, X_train, X_test, y_train, y_test
        )
        trained_models[model_name] = model
        all_metrics[model_name] = metrics
        predictions[model_name] = y_pred
    
    return trained_models, all_metrics, predictions


# ============================================================================
# MODEL EVALUATION & COMPARISON
# ============================================================================

def compare_models(all_metrics: dict) -> str:
    """Compare all models and return best model name."""
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON")
    logger.info("="*70)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df = comparison_df.sort_values('test_r2', ascending=False)
    
    logger.info("\nModel Performance Ranking (sorted by Test R²):")
    logger.info("\n" + comparison_df.to_string())
    
    # Select best model
    best_model = comparison_df.index[0]
    logger.info(f"\n🏆 BEST MODEL: {best_model}")
    logger.info(f"   Test RMSE: ${comparison_df.loc[best_model, 'test_rmse']:,.2f}")
    logger.info(f"   Test R²: {comparison_df.loc[best_model, 'test_r2']:.4f}")
    
    return best_model, comparison_df


# ============================================================================
# MLFLOW TRACKING
# ============================================================================

def log_models_to_mlflow(trained_models, all_metrics, best_model_name):
    """Log all models and metrics to MLflow."""
    logger.info("\n" + "="*70)
    logger.info("LOGGING TO MLFLOW")
    logger.info("="*70)
    
    # Setup MLflow
    mlflow.set_tracking_uri("./mlruns")
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"✅ Created new experiment: {MLFLOW_EXPERIMENT_NAME}")
    except:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        logger.info(f"✅ Using existing experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Log each model
    run_ids = {}
    for model_name, model in trained_models.items():
        with mlflow.start_run(run_name=model_name):
            # Log metrics
            for metric_name, value in all_metrics[model_name].items():
                mlflow.log_metric(metric_name, value)
            
            # Log parameters
            mlflow.log_param('model_name', model_name)
            mlflow.log_param('test_size', TEST_SIZE)
            mlflow.log_param('random_state', RANDOM_STATE)
            
            # Log model
            if model_name == 'XGBoost':
                mlflow.xgboost.log_model(model, 'model')
            else:
                mlflow.sklearn.log_model(model, 'model')
            
            # Store run ID
            run_ids[model_name] = mlflow.active_run().info.run_id
            
            logger.info(f"✅ Logged {model_name} to MLflow")
    
    return run_ids, experiment_id


def promote_best_model_to_production(best_model_name: str, run_ids: dict):
    """Register and promote best model to production."""
    logger.info("\n" + "="*70)
    logger.info("PROMOTING BEST MODEL TO PRODUCTION")
    logger.info("="*70)
    
    client = MlflowClient()
    best_run_id = run_ids[best_model_name]
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        # Register model
        result = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
        logger.info(f"✅ Model registered: {MODEL_REGISTRY_NAME}")
        logger.info(f"   Version: {result.version}")
        logger.info(f"   Run ID: {best_run_id}")
        
        # Set alias to production
        try:
            client.set_registered_model_alias(
                name=MODEL_REGISTRY_NAME,
                alias="production",
                version=result.version
            )
            logger.info(f"✅ Alias 'production' set to version {result.version}")
        except:
            # Fallback for older MLflow versions
            logger.info(f"⚠️ Could not set alias (older MLflow version)")
        
        return result.version
    except Exception as e:
        logger.warning(f"⚠️ Could not register model: {str(e)}")
        return None


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(data_validation: dict, comparison_df: pd.DataFrame, 
                   best_model: str, output_path: str):
    """Generate pipeline execution report."""
    logger.info("\n" + "="*70)
    logger.info("GENERATING REPORT")
    logger.info("="*70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_quality': {
            'total_rows': data_validation['total_rows'],
            'total_columns': data_validation['total_columns'],
            'duplicates': data_validation['duplicates']
        },
        'best_model': best_model,
        'model_metrics': comparison_df.to_dict(orient='index'),
        'pipeline_status': 'SUCCESS'
    }
    
    # Save report
    report_path = os.path.join(output_path, 'pipeline_report.json')
    os.makedirs(output_path, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Report saved to {report_path}")
    return report


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(data_path: str, output_path: str = './mlruns'):
    """Execute the complete MLOps pipeline."""
    logger.info("\n" + "="*70)
    logger.info("STARTING MLOPS PIPELINE")
    logger.info("="*70)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        # 1. Load data
        df = load_data(data_path)
        
        # 2. Validate data
        data_validation = validate_data(df)
        
        # 3. Preprocess data
        X_train, X_test, y_train, y_test, label_encoders, feature_names = preprocess_data(df)
        
        # 4. Train models
        trained_models, all_metrics, predictions = train_models(X_train, X_test, y_train, y_test)
        
        # 5. Compare models
        best_model, comparison_df = compare_models(all_metrics)
        
        # 6. Log to MLflow
        run_ids, experiment_id = log_models_to_mlflow(trained_models, all_metrics, best_model)
        
        # 7. Promote best model
        version = promote_best_model_to_production(best_model, run_ids)
        
        # 8. Generate report
        report = generate_report(data_validation, comparison_df, best_model, output_path)
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Best Model: {best_model}")
        logger.info(f"   MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
        logger.info(f"   Model Registry: {MODEL_REGISTRY_NAME} v{version}")
        logger.info(f"\n🌐 Access MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")
        
        return report
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Pipeline for House Price Prediction")
    parser.add_argument('--data-path', type=str, default='train.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--output-path', type=str, default='./mlruns',
                       help='Path for MLflow and reports output')
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(args.data_path, args.output_path)
