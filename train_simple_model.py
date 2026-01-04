"""
Script minimal pour entraîner les modèles et les sauvegarder localement
Utilise les mêmes 3 modèles que pipeline.py: LinearRegression, RandomForest, XGBoost
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle
import os
import json
from datetime import datetime

# Charger les données
train = pd.read_csv('train.csv')

# Séparer features et target
X = train.drop(['SalePrice', 'Id'], axis=1)
y = train['SalePrice']

# Preprocessing simple
# Colonnes numériques et catégorielles
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Imputation simple
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X[categorical_cols] = X[categorical_cols].fillna('Missing')

# Label encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print("🔄 Entraînement et comparaison de 3 modèles...")

# Définir les mêmes 3 modèles que pipeline.py
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective='reg:squarederror'
    )
}

# Entraîner et évaluer chaque modèle
results = {}
best_model_name = None
best_rmse = float('inf')
best_model = None

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X, y)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -cv_scores.mean()
    cv_std = cv_scores.std()
    r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    r2 = r2_scores.mean()
    
    is_best = (rmse < best_rmse)
    
    results[name] = {
        'test_rmse': round(rmse, 2),
        'test_r2': round(r2, 4),
        'cv_mean': round(r2, 4),
        'cv_std': round(cv_std, 4),
        'is_best': False  # Will update after finding best
    }
    
    print(f"    RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name
        best_model = model

# Mark the best model
results[best_model_name]['is_best'] = True

print(f"\n🏆 Meilleur modèle: {best_model_name} (RMSE: {best_rmse:.2f})")

# Sauvegarder le meilleur modèle
os.makedirs('model_artifacts', exist_ok=True)
with open('model_artifacts/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model_artifacts/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Sauvegarder les feature names
with open('model_artifacts/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Sauvegarder model_info.json
model_info = {
    'model_name': best_model_name,
    'model_type': type(best_model).__name__,
    'test_rmse': round(best_rmse, 2),
    'test_r2': results[best_model_name]['test_r2'],
    'n_features': len(X.columns),
    'n_samples': len(X),
    'timestamp': datetime.now().isoformat(),
    'version': '1.0.0'
}
with open('model_artifacts/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

# Sauvegarder comparison_results.json
comparison_results = {
    'best_model': best_model_name,
    'best_rmse': round(best_rmse, 2),
    'best_r2': results[best_model_name]['test_r2'],
    'timestamp': datetime.now().isoformat(),
    'models': results,
    'training_samples': len(X),
    'features_count': len(X.columns)
}
with open('model_artifacts/comparison_results.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print(f"\n✅ Modèle entraîné et sauvegardé dans model_artifacts/")
print(f"Features: {len(X.columns)}")
print(f"Samples: {len(X)}")
