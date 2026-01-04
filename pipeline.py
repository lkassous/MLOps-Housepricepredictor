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
        is_best = (model_name == best_model_name)
        
        # Create descriptive run name
        run_name = f"🏆 {model_name} (BEST)" if is_best else model_name
        
        with mlflow.start_run(run_name=run_name):
            # Log metrics
            for metric_name, value in all_metrics[model_name].items():
                mlflow.log_metric(metric_name, value)
            
            # Log parameters
            mlflow.log_param('model_name', model_name)
            mlflow.log_param('test_size', TEST_SIZE)
            mlflow.log_param('random_state', RANDOM_STATE)
            
            # Add tags to identify the best model
            mlflow.set_tag('model_type', model_name)
            mlflow.set_tag('is_best_model', str(is_best))
            mlflow.set_tag('pipeline_run', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            if is_best:
                mlflow.set_tag('status', 'production_candidate')
                mlflow.set_tag('selected_for_deployment', 'true')
                logger.info(f"\n{'='*70}")
                logger.info(f"🏆 BEST MODEL SELECTED: {model_name}")
                logger.info(f"   Test RMSE: ${all_metrics[model_name]['test_rmse']:,.2f}")
                logger.info(f"   Test R²: {all_metrics[model_name]['test_r2']:.4f}")
                logger.info(f"   CV R² (mean ± std): {all_metrics[model_name]['cv_mean']:.4f} ± {all_metrics[model_name]['cv_std']:.4f}")
                logger.info(f"{'='*70}\n")
            else:
                mlflow.set_tag('status', 'archived')
            
            # Log model - use sklearn for all models (more compatible)
            try:
                if model_name == 'XGBoost':
                    # Try xgboost flavor first, fallback to sklearn
                    try:
                        mlflow.xgboost.log_model(model, 'model')
                    except TypeError:
                        # Fallback for XGBoost compatibility issues
                        mlflow.sklearn.log_model(model, 'model')
                else:
                    mlflow.sklearn.log_model(model, 'model')
            except Exception as e:
                logger.warning(f"⚠️ Could not log model artifact: {e}")
            
            # Store run ID
            run_ids[model_name] = mlflow.active_run().info.run_id
            
            status_emoji = "🏆" if is_best else "📊"
            logger.info(f"{status_emoji} Logged {model_name} to MLflow (Run ID: {run_ids[model_name][:8]}...)")
    
    return run_ids, experiment_id


def promote_best_model_to_production(best_model_name: str, run_ids: dict):
    """Register and promote best model to production."""
    logger.info("\n" + "="*70)
    logger.info("🚀 PROMOTING BEST MODEL TO PRODUCTION")
    logger.info("="*70)
    
    client = MlflowClient()
    best_run_id = run_ids[best_model_name]
    model_uri = f"runs:/{best_run_id}/model"
    
    logger.info(f"\n📌 Selected Model: {best_model_name}")
    logger.info(f"📌 Run ID: {best_run_id}")
    logger.info(f"📌 Model URI: {model_uri}\n")
    
    try:
        # Register model
        result = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
        logger.info(f"\n✅ MODEL REGISTERED SUCCESSFULLY")
        logger.info(f"   Registry Name: {MODEL_REGISTRY_NAME}")
        logger.info(f"   Version: {result.version}")
        logger.info(f"   Model Type: {best_model_name}")
        logger.info(f"   Run ID: {best_run_id}")
        
        # Update model description
        try:
            client.update_registered_model(
                name=MODEL_REGISTRY_NAME,
                description=f"Production model for house price prediction. Current: {best_model_name} (v{result.version})"
            )
            client.update_model_version(
                name=MODEL_REGISTRY_NAME,
                version=result.version,
                description=f"Best model selected by pipeline: {best_model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            logger.debug(f"Could not update descriptions: {e}")
        
        # Set alias to production
        try:
            client.set_registered_model_alias(
                name=MODEL_REGISTRY_NAME,
                alias="production",
                version=result.version
            )
            logger.info(f"\n🎯 Production alias set to version {result.version}")
            logger.info(f"\n{'='*70}")
            logger.info(f"✨ {best_model_name} is now in PRODUCTION")
            logger.info(f"{'='*70}\n")
        except:
            # Fallback for older MLflow versions
            logger.info(f"⚠️ Could not set alias (older MLflow version)")
        
        return result.version
    except Exception as e:
        logger.warning(f"⚠️ Could not register model: {str(e)}")
        return None


# ============================================================================
# SAVE MODEL FOR DEPLOYMENT
# ============================================================================

def save_best_model_for_deployment(model, model_name: str, label_encoders: dict, 
                                    feature_names: list, comparison_df: pd.DataFrame):
    """Save the best model and artifacts for deployment."""
    import pickle
    
    logger.info("\n" + "="*70)
    logger.info("SAVING MODEL FOR DEPLOYMENT")
    logger.info("="*70)
    
    # Create model_artifacts directory
    os.makedirs('model_artifacts', exist_ok=True)
    
    # Save model
    model_path = 'model_artifacts/model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✅ Model saved: {model_path}")
    
    # Save label encoders
    encoders_path = 'model_artifacts/label_encoders.pkl'
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    logger.info(f"✅ Label encoders saved: {encoders_path}")
    
    # Save feature names
    features_path = 'model_artifacts/feature_names.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info(f"✅ Feature names saved: {features_path}")
    
    # Save model info
    model_info = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'timestamp': datetime.now().isoformat(),
        'feature_count': len(feature_names)
    }
    info_path = 'model_artifacts/model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"✅ Model info saved: {info_path}")
    
    # Save comparison results for the interface
    comparison_data = {
        'best_model': model_name,
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    for idx, row in comparison_df.iterrows():
        comparison_data['models'][idx] = {
            'test_r2': float(row['test_r2']),
            'test_rmse': float(row['test_rmse']),
            'test_mae': float(row['test_mae']),
            'cv_mean': float(row['cv_mean']),
            'cv_std': float(row['cv_std']),
            'is_best': idx == model_name
        }
    
    comparison_path = 'model_artifacts/comparison_results.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    logger.info(f"✅ Comparison results saved: {comparison_path}")
    
    logger.info(f"\n📦 All artifacts saved to model_artifacts/")


# ============================================================================
# REPORTING
# ============================================================================

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def generate_report(data_validation: dict, comparison_df: pd.DataFrame, 
                   best_model: str, output_path: str):
    """Generate pipeline execution report."""
    logger.info("\n" + "="*70)
    logger.info("GENERATING REPORT")
    logger.info("="*70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_quality': {
            'total_rows': int(data_validation['total_rows']),
            'total_columns': int(data_validation['total_columns']),
            'duplicates': int(data_validation['duplicates'])
        },
        'best_model': best_model,
        'model_metrics': convert_to_serializable(comparison_df.to_dict(orient='index')),
        'pipeline_status': 'SUCCESS'
    }
    
    # Save report in output_path
    report_path = os.path.join(output_path, 'pipeline_report.json')
    os.makedirs(output_path, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save report at root level for GitHub Actions
    root_report_path = 'pipeline_report.json'
    with open(root_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Report saved to {report_path}")
    logger.info(f"✅ Report also saved to {root_report_path}")
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
        
        # 8. Save best model to model_artifacts for deployment
        save_best_model_for_deployment(trained_models[best_model], best_model, 
                                       label_encoders, feature_names, comparison_df)
        
        # 9. Generate report
        report = generate_report(data_validation, comparison_df, best_model, output_path)
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Best Model: {best_model}")
        logger.info(f"   MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
        logger.info(f"   Model Registry: {MODEL_REGISTRY_NAME} v{version}")
        logger.info(f"   Model saved to: model_artifacts/")
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
