"""
MLflow Demo: Build and Compare Machine Learning Models
=======================================================
This script demonstrates:
1. Setting up MLflow tracking
2. Creating a machine learning pipeline
3. Training multiple models
4. Logging parameters, metrics, and models
5. Registering the best model
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Step 1: Set up MLflow
# ============================================
print("=" * 60)
print("Step 1: Setting up MLflow")
print("=" * 60)

# Set the tracking URI to local mlruns directory
mlflow.set_tracking_uri("./mlruns")

# Create or get experiment
experiment_name = "Iris-Classification-Demo"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"✅ Created new experiment: {experiment_name}")
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"✅ Using existing experiment: {experiment_name}")

mlflow.set_experiment(experiment_name)
print(f"   Experiment ID: {experiment_id}")

# ============================================
# Step 2: Create Machine Learning Pipeline
# ============================================
print("\n" + "=" * 60)
print("Step 2: Creating Machine Learning Pipeline")
print("=" * 60)

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"✅ Dataset loaded: Iris")
print(f"   Features: {feature_names}")
print(f"   Classes: {list(target_names)}")
print(f"   Samples: {len(X)}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Define models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, random_state=42),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
}

print(f"\n✅ Models to train: {list(models.keys())}")

# ============================================
# Step 3: Track Experiments with MLflow
# ============================================
print("\n" + "=" * 60)
print("Step 3: Training and Logging Models with MLflow")
print("=" * 60)

results = []

for model_name, model in models.items():
    print(f"\n🔄 Training {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Log parameters
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('test_size', 0.2)
        mlflow.log_param('random_state', 42)
        
        # Log model-specific parameters
        if hasattr(model, 'max_iter'):
            mlflow.log_param('max_iter', model.max_iter)
        if hasattr(model, 'n_estimators'):
            mlflow.log_param('n_estimators', model.n_estimators)
        if hasattr(model, 'max_depth') and model.max_depth is not None:
            mlflow.log_param('max_depth', model.max_depth)
        if hasattr(model, 'C'):
            mlflow.log_param('C', model.C)
        if hasattr(model, 'kernel'):
            mlflow.log_param('kernel', model.kernel)
        if hasattr(model, 'n_neighbors'):
            mlflow.log_param('n_neighbors', model.n_neighbors)
        
        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('cv_mean', cv_mean)
        mlflow.log_metric('cv_std', cv_std)
        
        # Log the model
        mlflow.sklearn.log_model(model, 'model')
        
        # Store results
        run_id = mlflow.active_run().info.run_id
        results.append({
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_mean,
            'run_id': run_id
        })
        
        print(f"   ✅ {model_name}")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      F1-Score: {f1:.4f}")
        print(f"      CV Mean:  {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"      Run ID:   {run_id[:8]}...")

# ============================================
# Step 4: Compare Models
# ============================================
print("\n" + "=" * 60)
print("Step 4: Model Comparison Summary")
print("=" * 60)

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('accuracy', ascending=False)

print("\n📊 Model Performance Ranking:")
print("-" * 70)
print(f"{'Rank':<6}{'Model':<20}{'Accuracy':<12}{'F1-Score':<12}{'CV Mean':<12}")
print("-" * 70)

for i, row in results_df.iterrows():
    rank = results_df.index.get_loc(i) + 1
    print(f"{rank:<6}{row['model_name']:<20}{row['accuracy']:<12.4f}{row['f1_score']:<12.4f}{row['cv_mean']:<12.4f}")

print("-" * 70)

# ============================================
# Step 5: Register Best Model
# ============================================
print("\n" + "=" * 60)
print("Step 5: Registering Best Model")
print("=" * 60)

# Get the best model
best_model = results_df.iloc[0]
best_run_id = best_model['run_id']
best_model_name = best_model['model_name']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {best_model['accuracy']:.4f}")
print(f"   Run ID: {best_run_id}")

# Register the model
model_registry_name = "IrisClassifier"
model_uri = f"runs:/{best_run_id}/model"

try:
    result = mlflow.register_model(model_uri, model_registry_name)
    print(f"\n✅ Model registered: {model_registry_name}")
    print(f"   Version: {result.version}")
    
    # Transition to Production
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_registry_name,
        version=result.version,
        stage="Production"
    )
    print(f"   Stage: Production")
except Exception as e:
    print(f"\n⚠️ Model registration: {str(e)[:50]}...")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("✅ MLflow Demo Complete!")
print("=" * 60)
print(f"""
📊 Summary:
   - Experiment: {experiment_name}
   - Models trained: {len(models)}
   - Best model: {best_model_name} (Accuracy: {best_model['accuracy']:.4f})
   
🌐 View results in MLflow UI:
   URL: http://localhost:5000
   
📁 Experiment location: ./mlruns
""")
