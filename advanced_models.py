"""
Test de nouveaux algorithmes: LightGBM et CatBoost
avec tracking MLflow
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House-Prices-Advanced-Models")

print("=" * 70)
print("ENTRAÎNEMENT DE MODÈLES AVANCÉS")
print("=" * 70)

# 1. Chargement et préparation des données
print("\n1. Chargement des données...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Sauvegarder l'ID du test set
test_ids = test_df['Id']

# Supprimer la colonne Id
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. Préparation des données (même preprocessing que précédemment)
print("\n2. Préparation des données...")

# Séparer X et y
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# Identifier les colonnes numériques et catégorielles
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Features numériques: {len(numeric_features)}")
print(f"Features catégorielles: {len(categorical_features)}")

# Remplir les valeurs manquantes
for col in numeric_features:
    X[col].fillna(X[col].median(), inplace=True)
    test_df[col].fillna(test_df[col].median(), inplace=True)

for col in categorical_features:
    X[col].fillna(X[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# Encoder les variables catégorielles
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 3. Entraîner LightGBM
print("\n" + "=" * 70)
print("3. ENTRAÎNEMENT LIGHTGBM")
print("=" * 70)

try:
    import lightgbm as lgb
    
    with mlflow.start_run(run_name="LightGBM-Default"):
        print("\nEntraînement en cours...")
        
        # Paramètres
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 1000
        }
        
        # Log des paramètres
        mlflow.log_params(params)
        
        # Entraînement
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_test, y_test)],
                 eval_metric='rmse',
                 callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        # Prédictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Métriques Train
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Métriques Test
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log des métriques
        mlflow.log_metrics({
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        })
        
        # Log du modèle
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\nRésultats LightGBM:")
        print(f"  Train RMSE: ${train_rmse:,.2f}")
        print(f"  Test RMSE: ${test_rmse:,.2f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test MAE: ${test_mae:,.2f}")
        
        lgb_model = model
        lgb_rmse = test_rmse
        
except ImportError:
    print("\nLightGBM non installé. Installation...")
    import subprocess
    subprocess.run(['pip', 'install', 'lightgbm'], check=True)
    print("Veuillez réexécuter le script après l'installation.")
    lgb_model = None
    lgb_rmse = float('inf')

# 4. Entraîner CatBoost
print("\n" + "=" * 70)
print("4. ENTRAÎNEMENT CATBOOST")
print("=" * 70)

try:
    from catboost import CatBoostRegressor
    
    with mlflow.start_run(run_name="CatBoost-Default"):
        print("\nEntraînement en cours...")
        
        # Paramètres
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50
        }
        
        # Log des paramètres
        mlflow.log_params(params)
        
        # Entraînement
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, 
                 eval_set=(X_test, y_test),
                 verbose=False)
        
        # Prédictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Métriques Train
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Métriques Test
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log des métriques
        mlflow.log_metrics({
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        })
        
        # Log du modèle
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\nRésultats CatBoost:")
        print(f"  Train RMSE: ${train_rmse:,.2f}")
        print(f"  Test RMSE: ${test_rmse:,.2f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test MAE: ${test_mae:,.2f}")
        
        cat_model = model
        cat_rmse = test_rmse
        
except ImportError:
    print("\nCatBoost non installé. Installation...")
    import subprocess
    subprocess.run(['pip', 'install', 'catboost'], check=True)
    print("Veuillez réexécuter le script après l'installation.")
    cat_model = None
    cat_rmse = float('inf')

# 5. Comparaison finale
print("\n" + "=" * 70)
print("COMPARAISON DES MODÈLES")
print("=" * 70)

print("\nMéthodes disponibles:")
print(f"  - LightGBM: {'✓' if lgb_model else '✗'} (RMSE: ${lgb_rmse:,.2f})")
print(f"  - CatBoost: {'✓' if cat_model else '✗'} (RMSE: ${cat_rmse:,.2f})")

print("\n" + "=" * 70)
print("ENTRAÎNEMENT TERMINÉ")
print("=" * 70)
print("\nConsultez MLflow UI pour comparer tous les modèles:")
print("http://127.0.0.1:5000")
