"""
Feature Engineering Avancé pour améliorer les prédictions
Création de nouvelles features: interactions, polynomiales, agrégations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House-Prices-Feature-Engineering")

print("=" * 70)
print("FEATURE ENGINEERING AVANCÉ")
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

# Combiner pour le feature engineering
combined = pd.concat([X, test_df], axis=0, ignore_index=True)
print(f"Combined shape: {combined.shape}")

# 2. Feature Engineering
print("\n2. Création de nouvelles features...")

# 2.1 Features de surface totale
print("  - Features de surface...")
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
combined['TotalBathrooms'] = combined['FullBath'] + 0.5 * combined['HalfBath'] + \
                              combined['BsmtFullBath'] + 0.5 * combined['BsmtHalfBath']
combined['TotalPorchSF'] = combined['OpenPorchSF'] + combined['EnclosedPorch'] + \
                           combined['3SsnPorch'] + combined['ScreenPorch']

# 2.2 Features d'âge et de rénovation
print("  - Features d'âge...")
combined['HouseAge'] = combined['YrSold'] - combined['YearBuilt']
combined['RemodAge'] = combined['YrSold'] - combined['YearRemodAdd']
combined['IsRemodeled'] = (combined['YearBuilt'] != combined['YearRemodAdd']).astype(int)
combined['YearsSinceRemod'] = combined['YrSold'] - combined['YearRemodAdd']

# 2.3 Features booléennes
print("  - Features booléennes...")
combined['HasPool'] = (combined['PoolArea'] > 0).astype(int)
combined['Has2ndFloor'] = (combined['2ndFlrSF'] > 0).astype(int)
combined['HasGarage'] = (combined['GarageArea'] > 0).astype(int)
combined['HasBsmt'] = (combined['TotalBsmtSF'] > 0).astype(int)
combined['HasFireplace'] = (combined['Fireplaces'] > 0).astype(int)

# 2.4 Features de qualité combinée
print("  - Features de qualité...")
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}

# Convertir les qualités en numériques
for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
            'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
    combined[col] = combined[col].fillna('NA').map(quality_map)

combined['OverallScore'] = combined['OverallQual'] * combined['OverallCond']
combined['ExterScore'] = combined['ExterQual'] * combined['ExterCond']
combined['BsmtScore'] = combined['BsmtQual'] * combined['BsmtCond']
combined['GarageScore'] = combined['GarageQual'] * combined['GarageCond']

# 2.5 Features de ratio
print("  - Features de ratio...")
combined['LivingAreaRatio'] = combined['GrLivArea'] / (combined['LotArea'] + 1)
combined['GarageRatio'] = combined['GarageArea'] / (combined['GrLivArea'] + 1)
combined['BsmtRatio'] = combined['TotalBsmtSF'] / (combined['GrLivArea'] + 1)

# 2.6 Features d'interaction
print("  - Features d'interaction...")
combined['QualityArea'] = combined['OverallQual'] * combined['GrLivArea']
combined['QualityBaths'] = combined['OverallQual'] * combined['TotalBathrooms']
combined['QualityGarage'] = combined['OverallQual'] * combined['GarageArea']

# 2.7 Features de quartier (agrégations)
print("  - Features de quartier...")
neighborhood_features = combined.groupby('Neighborhood').agg({
    'GrLivArea': 'mean',
    'OverallQual': 'mean',
    'YearBuilt': 'mean'
}).add_prefix('Neighborhood_')

combined = combined.join(neighborhood_features, on='Neighborhood')

# 2.8 Features polynomiales pour variables importantes
print("  - Features polynomiales...")
combined['GrLivArea_Squared'] = combined['GrLivArea'] ** 2
combined['GrLivArea_Cubed'] = combined['GrLivArea'] ** 3
combined['OverallQual_Squared'] = combined['OverallQual'] ** 2
combined['TotalSF_Squared'] = combined['TotalSF'] ** 2

# 3. Gestion des valeurs manquantes
print("\n3. Gestion des valeurs manquantes...")

numeric_features = combined.select_dtypes(include=['int64', 'float64']).columns
categorical_features = combined.select_dtypes(include=['object']).columns

for col in numeric_features:
    combined[col].fillna(combined[col].median(), inplace=True)

for col in categorical_features:
    combined[col].fillna(combined[col].mode()[0] if len(combined[col].mode()) > 0 else 'None', inplace=True)

# 4. Encodage des variables catégorielles
print("\n4. Encodage des variables catégorielles...")

for col in categorical_features:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# 5. Séparation train/test
print("\n5. Séparation des données...")

n_train = len(X)
X_engineered = combined[:n_train]
test_engineered = combined[n_train:]

print(f"Features originales: {X.shape[1]}")
print(f"Features après engineering: {X_engineered.shape[1]}")
print(f"Nouvelles features créées: {X_engineered.shape[1] - X.shape[1]}")

# Split train/validation
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

# 6. Entraînement du modèle avec nouvelles features
print("\n" + "=" * 70)
print("6. ENTRAÎNEMENT AVEC FEATURES ENGINEERED")
print("=" * 70)

with mlflow.start_run(run_name="XGBoost-Feature-Engineering"):
    print("\nEntraînement en cours...")
    
    # Paramètres optimaux du tuning précédent
    params = {
        'n_estimators': 300,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:squarederror'
    }
    
    # Log des paramètres
    mlflow.log_params(params)
    mlflow.log_param('n_features', X_engineered.shape[1])
    mlflow.log_param('feature_engineering', 'advanced')
    
    # Entraînement
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
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
    
    print(f"\nRésultats avec Feature Engineering:")
    print(f"  Train RMSE: ${train_rmse:,.2f}, MAE: ${train_mae:,.2f}, R²: {train_r2:.4f}")
    print(f"  Test RMSE: ${test_rmse:,.2f}, MAE: ${test_mae:,.2f}, R²: {test_r2:.4f}")
    
    # Feature importance
    print("\n7. Top 15 Features les plus importantes:")
    feature_importance = pd.DataFrame({
        'feature': X_engineered.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))
    
    # Sauvegarder feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    
    # 8. Générer prédictions pour soumission
    print("\n8. Génération des prédictions pour soumission...")
    test_predictions = model.predict(test_engineered)
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_predictions
    })
    
    submission.to_csv('submission_feature_engineering.csv', index=False)
    print("Fichier créé: submission_feature_engineering.csv")
    
    print(f"\nStatistiques des prédictions:")
    print(f"  Min: ${test_predictions.min():,.2f}")
    print(f"  Max: ${test_predictions.max():,.2f}")
    print(f"  Mean: ${test_predictions.mean():,.2f}")
    print(f"  Median: ${np.median(test_predictions):,.2f}")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING TERMINÉ")
print("=" * 70)
print("\nConsultez MLflow UI pour comparer avec les modèles précédents:")
print("http://127.0.0.1:5000")
