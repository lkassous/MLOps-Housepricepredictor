"""
Script minimal pour entraîner un modèle XGBoost et le sauvegarder localement
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import pickle
import os

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

# Entraîner XGBoost avec les meilleurs hyperparamètres
model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X, y)

# Sauvegarder le modèle
os.makedirs('model_artifacts', exist_ok=True)
with open('model_artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_artifacts/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Sauvegarder les feature names
with open('model_artifacts/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print(f"✅ Modèle entraîné et sauvegardé dans model_artifacts/")
print(f"Features: {len(X.columns)}")
print(f"Samples: {len(X)}")
