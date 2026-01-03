"""
Hyperparameter Tuning avec MLflow
Next Step du Tutorial: Perform hyperparameter tuning
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
EXPERIMENT_NAME = "House-Prices-Hyperparameter-Tuning"
MODEL_NAME = "HousePrices-TunedModel"

def load_and_prepare_data():
    """Charge et prépare les données"""
    print("\n📊 Chargement et préparation des données...")
    
    # Charger les données
    df = pd.read_csv('train.csv')
    
    # Supprimer les colonnes avec trop de valeurs manquantes
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    
    # Séparer features et target
    X = df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
    y = df['SalePrice']
    
    # Identifier les colonnes catégorielles et numériques
    categorical_features = X.select_dtypes(include=['object']).columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Remplir les valeurs manquantes
    for col in numeric_features:
        X[col].fillna(X[col].median(), inplace=True)
    
    for col in categorical_features:
        X[col].fillna(X[col].mode()[0], inplace=True)
    
    # Encoder les variables catégorielles
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Données chargées: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"✓ Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Évalue le modèle et retourne les métriques"""
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métriques
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    return metrics

def hyperparameter_tuning_xgboost(X_train, X_test, y_train, y_test):
    """Hyperparameter tuning pour XGBoost avec GridSearch"""
    
    print("\n" + "="*70)
    print("🔧 HYPERPARAMETER TUNING - XGBoost")
    print("="*70)
    
    # Définir la grille de paramètres
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    print(f"\n📋 Grille de paramètres:")
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n🔢 Total de combinaisons à tester: {total_combinations}")
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Modèle de base
    base_model = xgb.XGBRegressor(random_state=42)
    
    # GridSearchCV
    print(f"\n⏳ Recherche en cours (cela peut prendre quelques minutes)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Meilleurs paramètres
    best_params = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)
    
    print("\n" + "="*70)
    print("🏆 MEILLEURS PARAMÈTRES TROUVÉS")
    print("="*70)
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"\n   CV RMSE: ${best_score:,.2f}")
    print("="*70)
    
    # Entraîner le modèle final avec les meilleurs paramètres
    print("\n📈 Entraînement du modèle final avec les meilleurs paramètres...")
    
    with mlflow.start_run(run_name="XGBoost-Tuned-Best"):
        # Créer et entraîner le modèle
        best_model = xgb.XGBRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        # Évaluer
        metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)
        
        # Logger les paramètres
        mlflow.log_params(best_params)
        mlflow.log_param('model_name', 'XGBoost-Tuned')
        
        # Logger les métriques
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Logger le modèle (utiliser sklearn au lieu de xgboost pour éviter les erreurs de compatibilité)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Sauvegarder le run_id
        run_id = mlflow.active_run().info.run_id
        
        print("\n✅ Modèle entraîné et loggé dans MLflow")
        print(f"   Run ID: {run_id}")
    
    # Logger tous les résultats du grid search
    print("\n📊 Logging de tous les résultats du Grid Search...")
    
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    for idx, row in results_df.iterrows():
        with mlflow.start_run(run_name=f"XGBoost-GridSearch-{idx+1}", nested=True):
            # Logger les paramètres testés
            params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_')}
            mlflow.log_params(params)
            
            # Logger les métriques CV
            mlflow.log_metric('cv_rmse', np.sqrt(-row['mean_test_score']))
            mlflow.log_metric('cv_rmse_std', row['std_test_score'])
    
    print(f"✓ {len(results_df)} runs loggés")
    
    return best_model, metrics, best_params, run_id

def register_tuned_model(run_id, metrics, best_params):
    """Enregistre le modèle tuné dans le Model Registry"""
    
    print("\n" + "="*70)
    print("📝 ENREGISTREMENT DU MODÈLE TUNÉ")
    print("="*70)
    
    client = MlflowClient()
    
    # URI du modèle
    model_uri = f"runs:/{run_id}/model"
    
    try:
        # Créer ou récupérer le modèle enregistré
        try:
            registered_model = client.get_registered_model(MODEL_NAME)
            print(f"✓ Modèle '{MODEL_NAME}' existe déjà")
        except:
            registered_model = client.create_registered_model(
                MODEL_NAME,
                description="Modèle XGBoost avec hyperparamètres optimisés par GridSearch"
            )
            print(f"✓ Modèle '{MODEL_NAME}' créé")
        
        # Créer une nouvelle version
        model_version = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run_id,
            description=f"XGBoost optimisé - Test RMSE: ${metrics['test_rmse']:,.2f}, R²: {metrics['test_r2']:.4f}"
        )
        
        version_number = model_version.version
        print(f"✓ Version {version_number} créée")
        
        # Transition vers Production
        print(f"\n🚀 Transition vers Production...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_number,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"✓ Version {version_number} est en Production")
        
        # Ajouter des tags
        client.set_model_version_tag(MODEL_NAME, version_number, "optimization", "GridSearch")
        client.set_model_version_tag(MODEL_NAME, version_number, "framework", "xgboost")
        
        print("\n" + "="*70)
        print("✅ MODÈLE TUNÉ ENREGISTRÉ AVEC SUCCÈS!")
        print("="*70)
        print(f"Nom: {MODEL_NAME}")
        print(f"Version: {version_number}")
        print(f"Test RMSE: ${metrics['test_rmse']:,.2f}")
        print(f"Test MAE: ${metrics['test_mae']:,.2f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"\n🔗 MLflow UI: http://127.0.0.1:5000/#/models/{MODEL_NAME}")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

def main():
    """Fonction principale"""
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - NEXT STEP DU TUTORIAL")
    print("="*70)
    print("\n🎯 Objectif: Optimiser les hyperparamètres de XGBoost avec GridSearch")
    print("   et logger tous les résultats dans MLflow")
    
    # Charger les données
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Hyperparameter tuning
    best_model, metrics, best_params, run_id = hyperparameter_tuning_xgboost(
        X_train, X_test, y_train, y_test
    )
    
    # Afficher les résultats
    print("\n" + "="*70)
    print("📊 RÉSULTATS FINAUX")
    print("="*70)
    print(f"Train RMSE: ${metrics['train_rmse']:,.2f}")
    print(f"Test RMSE:  ${metrics['test_rmse']:,.2f}")
    print(f"Train MAE:  ${metrics['train_mae']:,.2f}")
    print(f"Test MAE:   ${metrics['test_mae']:,.2f}")
    print(f"Train R²:   {metrics['train_r2']:.4f}")
    print(f"Test R²:    {metrics['test_r2']:.4f}")
    print("="*70)
    
    # Enregistrer le modèle
    register_tuned_model(run_id, metrics, best_params)
    
    print("\n✅ Processus terminé!")
    print("\n📌 Prochaines étapes:")
    print("   1. Ouvrez http://127.0.0.1:5000")
    print("   2. Comparez l'expérience 'House-Prices-Hyperparameter-Tuning'")
    print("   3. Consultez le Model Registry pour voir le modèle optimisé")

if __name__ == "__main__":
    main()
