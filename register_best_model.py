"""
Script pour enregistrer le meilleur modèle dans MLflow Model Registry
Étape 5 du Tutorial MLflow
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Configuration
EXPERIMENT_NAME = "House-Prices-Regression"
MODEL_NAME = "HousePrices-BestModel"

def get_best_run():
    """Trouve le meilleur run basé sur le RMSE le plus bas"""
    client = MlflowClient()
    
    # Récupérer l'expérience
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"❌ Expérience '{EXPERIMENT_NAME}' non trouvée")
        return None
    
    # Rechercher tous les runs de l'expérience
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_rmse ASC"],  # Trier par RMSE croissant
        max_results=1
    )
    
    if not runs:
        print("❌ Aucun run trouvé")
        return None
    
    best_run = runs[0]
    return best_run

def register_model(run):
    """Enregistre le modèle dans le Model Registry"""
    client = MlflowClient()
    
    # Informations du run
    run_id = run.info.run_id
    model_name = run.data.params.get('model_name', 'Unknown')
    rmse = run.data.metrics.get('test_rmse', 0)
    mae = run.data.metrics.get('test_mae', 0)
    r2 = run.data.metrics.get('test_r2', 0)
    
    print("\n" + "="*70)
    print("🏆 MEILLEUR MODÈLE TROUVÉ")
    print("="*70)
    print(f"Modèle: {model_name}")
    print(f"Run ID: {run_id}")
    print(f"Test RMSE: ${rmse:,.2f}")
    print(f"Test MAE: ${mae:,.2f}")
    print(f"Test R²: {r2:.4f}")
    print("="*70)
    
    # URI du modèle
    model_uri = f"runs:/{run_id}/model"
    
    # Enregistrer le modèle
    print(f"\n📝 Enregistrement du modèle '{MODEL_NAME}' dans le Model Registry...")
    
    try:
        # Vérifier si le modèle existe déjà
        try:
            registered_model = client.get_registered_model(MODEL_NAME)
            print(f"✓ Modèle '{MODEL_NAME}' existe déjà")
        except:
            # Créer le modèle s'il n'existe pas
            registered_model = client.create_registered_model(
                MODEL_NAME,
                description=f"Meilleur modèle pour la prédiction des prix de maisons. "
                           f"Entraîné avec {model_name}."
            )
            print(f"✓ Modèle '{MODEL_NAME}' créé")
        
        # Créer une nouvelle version
        model_version = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run_id,
            description=f"{model_name} - RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, R²: {r2:.4f}"
        )
        
        version_number = model_version.version
        print(f"✓ Version {version_number} créée")
        
        # Transition vers Production
        print(f"\n🚀 Transition de la version {version_number} vers 'Production'...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_number,
            stage="Production",
            archive_existing_versions=True  # Archive les anciennes versions en production
        )
        print(f"✓ Version {version_number} est maintenant en Production")
        
        # Ajouter des tags
        client.set_model_version_tag(MODEL_NAME, version_number, "framework", "scikit-learn/xgboost")
        client.set_model_version_tag(MODEL_NAME, version_number, "dataset", "house-prices")
        
        print("\n" + "="*70)
        print("✅ MODÈLE ENREGISTRÉ AVEC SUCCÈS!")
        print("="*70)
        print(f"Nom: {MODEL_NAME}")
        print(f"Version: {version_number}")
        print(f"Stage: Production")
        print(f"\n🔗 Voir dans MLflow UI: http://127.0.0.1:5000/#/models/{MODEL_NAME}")
        print("="*70)
        
        return model_version
        
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement: {e}")
        return None

def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print("ENREGISTREMENT DU MEILLEUR MODÈLE - ÉTAPE 5 DU TUTORIAL")
    print("="*70)
    
    # Trouver le meilleur run
    print(f"\n🔍 Recherche du meilleur modèle dans '{EXPERIMENT_NAME}'...")
    best_run = get_best_run()
    
    if best_run is None:
        print("\n❌ Impossible de trouver le meilleur modèle")
        return
    
    # Enregistrer le modèle
    model_version = register_model(best_run)
    
    if model_version:
        print("\n✅ Processus terminé avec succès!")
        print("\nProchaines étapes:")
        print("1. Ouvrez http://127.0.0.1:5000/#/models")
        print("2. Cliquez sur 'HousePrices-BestModel'")
        print("3. Vous verrez le modèle en Production avec toutes ses métriques")
        print("\n🎯 Vous pouvez maintenant déployer ce modèle ou l'utiliser pour des prédictions")
    else:
        print("\n❌ Échec de l'enregistrement du modèle")

if __name__ == "__main__":
    main()
