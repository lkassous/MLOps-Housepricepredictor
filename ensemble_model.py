"""
Ensemble Model - Stacking de plusieurs modèles
Combine XGBoost, LightGBM, et CatBoost pour de meilleures prédictions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House-Prices-Ensemble-Models")

print("=" * 70)
print("MODÈLE ENSEMBLE - STACKING")
print("=" * 70)

# 1. Chargement des données
print("\n1. Chargement des données...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

test_ids = test_df['Id']
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

# Séparer X et y
y = train_df['SalePrice']
X = train_df.drop('SalePrice', axis=1)

# 2. Préparation des données
print("\n2. Préparation des données...")

# Identifier les colonnes
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Remplir les valeurs manquantes
for col in numeric_features:
    X[col].fillna(X[col].median(), inplace=True)
    test_df[col].fillna(test_df[col].median(), inplace=True)

for col in categorical_features:
    X[col].fillna(X[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# Encoder les variables catégorielles
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 3. Définir les modèles de base
print("\n3. Définition des modèles de base...")

base_models = []

# XGBoost
print("  - XGBoost")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
base_models.append(('XGBoost', xgb_model))

# LightGBM
try:
    import lightgbm as lgb
    print("  - LightGBM")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        n_estimators=1000
    )
    base_models.append(('LightGBM', lgb_model))
except ImportError:
    print("  - LightGBM non disponible (installation requise)")

# CatBoost
try:
    from catboost import CatBoostRegressor
    print("  - CatBoost")
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    base_models.append(('CatBoost', cat_model))
except ImportError:
    print("  - CatBoost non disponible (installation requise)")

print(f"\nNombre de modèles de base: {len(base_models)}")

# 4. Stacking avec K-Fold Cross-Validation
print("\n" + "=" * 70)
print("4. STACKING - ENTRAÎNEMENT DES MODÈLES DE BASE")
print("=" * 70)

with mlflow.start_run(run_name="Ensemble-Stacking"):
    
    # Log des paramètres
    mlflow.log_param('n_base_models', len(base_models))
    mlflow.log_param('meta_model', 'Ridge')
    mlflow.log_param('cv_folds', 5)
    
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Matrices pour stocker les prédictions out-of-fold
    train_meta_features = np.zeros((X_train.shape[0], len(base_models)))
    test_meta_features = np.zeros((X_test.shape[0], len(base_models)))
    test_final_meta = np.zeros((test_df.shape[0], len(base_models)))
    
    # Pour chaque modèle de base
    for idx, (name, model) in enumerate(base_models):
        print(f"\n  Entraînement {name}...")
        
        test_fold_preds = np.zeros((X_test.shape[0], n_folds))
        test_final_preds = np.zeros((test_df.shape[0], n_folds))
        
        # K-Fold CV
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Entraîner le modèle
            if name == 'LightGBM':
                model.fit(X_tr, y_tr, 
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            elif name == 'CatBoost':
                model.fit(X_tr, y_tr,
                         eval_set=(X_val, y_val),
                         verbose=False)
            else:
                model.fit(X_tr, y_tr)
            
            # Prédictions out-of-fold
            train_meta_features[val_idx, idx] = model.predict(X_val)
            
            # Prédictions sur test set
            test_fold_preds[:, fold] = model.predict(X_test)
            test_final_preds[:, fold] = model.predict(test_df)
        
        # Moyenne des prédictions sur les folds
        test_meta_features[:, idx] = test_fold_preds.mean(axis=1)
        test_final_meta[:, idx] = test_final_preds.mean(axis=1)
        
        # Calculer RMSE pour ce modèle
        model_rmse = np.sqrt(mean_squared_error(y_train, train_meta_features[:, idx]))
        print(f"    RMSE (out-of-fold): ${model_rmse:,.2f}")
        mlflow.log_metric(f'{name}_oof_rmse', model_rmse)
    
    # 5. Entraîner le méta-modèle
    print("\n" + "=" * 70)
    print("5. ENTRAÎNEMENT DU MÉTA-MODÈLE (Ridge)")
    print("=" * 70)
    
    meta_model = Ridge(alpha=10.0)
    meta_model.fit(train_meta_features, y_train)
    
    # Prédictions finales
    y_train_pred = meta_model.predict(train_meta_features)
    y_test_pred = meta_model.predict(test_meta_features)
    
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
    
    print(f"\nRésultats Ensemble Stacking:")
    print(f"  Train RMSE: ${train_rmse:,.2f}, MAE: ${train_mae:,.2f}, R²: {train_r2:.4f}")
    print(f"  Test RMSE: ${test_rmse:,.2f}, MAE: ${test_mae:,.2f}, R²: {test_r2:.4f}")
    
    # Poids du méta-modèle
    print(f"\nPoids des modèles de base:")
    for idx, (name, _) in enumerate(base_models):
        print(f"  {name}: {meta_model.coef_[idx]:.4f}")
        mlflow.log_metric(f'{name}_weight', meta_model.coef_[idx])
    
    # Log du méta-modèle
    mlflow.sklearn.log_model(meta_model, "meta_model")
    
    # 6. Générer prédictions finales
    print("\n6. Génération des prédictions finales...")
    
    final_predictions = meta_model.predict(test_final_meta)
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': final_predictions
    })
    
    submission.to_csv('submission_ensemble.csv', index=False)
    print("Fichier créé: submission_ensemble.csv")
    
    print(f"\nStatistiques des prédictions:")
    print(f"  Min: ${final_predictions.min():,.2f}")
    print(f"  Max: ${final_predictions.max():,.2f}")
    print(f"  Mean: ${final_predictions.mean():,.2f}")
    print(f"  Median: ${np.median(final_predictions):,.2f}")

print("\n" + "=" * 70)
print("ENSEMBLE STACKING TERMINÉ")
print("=" * 70)
print("\nConsultez MLflow UI pour comparer tous les modèles:")
print("http://127.0.0.1:5000")
