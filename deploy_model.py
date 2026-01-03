"""
Déploiement et prédiction avec le modèle MLflow
Next Step du Tutorial: Deploy models using MLflow's deployment tools
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "HousePrices-BestModel"

def load_production_model():
    """Charge le modèle en Production depuis le Model Registry"""
    
    print("\n" + "="*70)
    print("🔄 CHARGEMENT DU MODÈLE EN PRODUCTION")
    print("="*70)
    
    try:
        # Charger le modèle en Production
        model_uri = f"models:/{MODEL_NAME}/Production"
        print(f"\n📦 Chargement du modèle: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Récupérer les informations du modèle
        client = MlflowClient()
        model_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        
        print(f"✓ Modèle chargé avec succès")
        print(f"   Version: {model_version.version}")
        print(f"   Run ID: {model_version.run_id}")
        print(f"   Stage: Production")
        print("="*70)
        
        return model, model_version
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("\n💡 Assurez-vous qu'un modèle est enregistré en Production")
        print("   Exécutez 'register_best_model.py' d'abord")
        return None, None

def prepare_test_data():
    """Prépare les données de test pour les prédictions"""
    
    print("\n📊 Préparation des données de test...")
    
    try:
        # Charger les données d'entraînement pour récupérer les transformations
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')
        
        # Sauvegarder les IDs pour la soumission
        test_ids = df_test['Id'].copy()
        
        # Supprimer les colonnes avec trop de valeurs manquantes (même traitement que l'entraînement)
        threshold = len(df_train) * 0.5
        cols_to_keep = df_train.dropna(thresh=threshold, axis=1).columns
        cols_to_keep = [col for col in cols_to_keep if col in df_test.columns and col != 'SalePrice']
        
        df_test = df_test[cols_to_keep]
        
        # Supprimer la colonne Id
        df_test = df_test.drop(['Id'], axis=1, errors='ignore')
        
        # Identifier les colonnes catégorielles et numériques
        categorical_features = df_test.select_dtypes(include=['object']).columns
        numeric_features = df_test.select_dtypes(include=['int64', 'float64']).columns
        
        # Remplir les valeurs manquantes
        for col in numeric_features:
            df_test[col].fillna(df_test[col].median(), inplace=True)
        
        for col in categorical_features:
            df_test[col].fillna(df_test[col].mode()[0] if len(df_test[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Encoder les variables catégorielles
        for col in categorical_features:
            le = LabelEncoder()
            # Fit sur train et test combinés pour éviter les erreurs
            combined = pd.concat([df_train[col].astype(str), df_test[col].astype(str)])
            le.fit(combined)
            df_test[col] = le.transform(df_test[col].astype(str))
        
        print(f"✓ Données de test préparées: {df_test.shape[0]} maisons, {df_test.shape[1]} features")
        
        return df_test, test_ids
        
    except FileNotFoundError:
        print("❌ Fichier test.csv non trouvé")
        return None, None
    except Exception as e:
        print(f"❌ Erreur lors de la préparation des données: {e}")
        return None, None

def make_predictions(model, X_test, test_ids):
    """Fait des prédictions sur les données de test"""
    
    print("\n" + "="*70)
    print("🔮 PRÉDICTIONS SUR LES DONNÉES DE TEST")
    print("="*70)
    
    try:
        # Faire les prédictions
        print(f"\n⏳ Prédiction en cours sur {len(X_test)} maisons...")
        predictions = model.predict(X_test)
        
        print(f"✓ Prédictions terminées")
        
        # Statistiques des prédictions
        print("\n📊 Statistiques des prédictions:")
        print(f"   Prix minimum prédit: ${predictions.min():,.2f}")
        print(f"   Prix maximum prédit: ${predictions.max():,.2f}")
        print(f"   Prix moyen prédit: ${predictions.mean():,.2f}")
        print(f"   Prix médian prédit: ${np.median(predictions):,.2f}")
        
        # Créer le DataFrame de soumission
        submission = pd.DataFrame({
            'Id': test_ids,
            'SalePrice': predictions
        })
        
        # Sauvegarder
        output_file = 'my_submission.csv'
        submission.to_csv(output_file, index=False)
        
        print(f"\n💾 Fichier de soumission créé: {output_file}")
        print(f"   Format: Id, SalePrice")
        print(f"   Nombre de lignes: {len(submission)}")
        
        # Afficher quelques exemples
        print("\n📋 Aperçu des premières prédictions:")
        print(submission.head(10).to_string(index=False))
        
        print("="*70)
        
        return submission
        
    except Exception as e:
        print(f"❌ Erreur lors des prédictions: {e}")
        return None

def serve_model_info():
    """Affiche les informations pour servir le modèle"""
    
    print("\n" + "="*70)
    print("🚀 DÉPLOIEMENT DU MODÈLE")
    print("="*70)
    
    print("\n📌 Options de déploiement MLflow:")
    
    print("\n1️⃣  Servir le modèle comme API REST:")
    print("   Commande:")
    print(f"   mlflow models serve -m models:/{MODEL_NAME}/Production -p 5001")
    print("   ")
    print("   Le modèle sera accessible sur http://127.0.0.1:5001")
    print("   Endpoint de prédiction: POST http://127.0.0.1:5001/invocations")
    
    print("\n2️⃣  Servir avec Docker:")
    print("   Commandes:")
    print(f"   mlflow models build-docker -m models:/{MODEL_NAME}/Production -n house-prices-model")
    print("   docker run -p 5001:8080 house-prices-model")
    
    print("\n3️⃣  Exporter le modèle:")
    print("   Python:")
    print(f"   model = mlflow.pyfunc.load_model('models:/{MODEL_NAME}/Production')")
    print("   predictions = model.predict(X_test)")
    
    print("\n4️⃣  Tester l'API (après avoir lancé le serveur):")
    print("   curl -X POST -H 'Content-Type: application/json' \\")
    print("        -d '{\"dataframe_split\": {\"columns\": [...], \"data\": [[...]]}}' \\")
    print("        http://127.0.0.1:5001/invocations")
    
    print("="*70)

def main():
    """Fonction principale"""
    
    print("\n" + "="*70)
    print("DÉPLOIEMENT ET PRÉDICTION - NEXT STEP DU TUTORIAL")
    print("="*70)
    print("\n🎯 Objectif: Charger le modèle Production et faire des prédictions")
    
    # Charger le modèle
    model, model_version = load_production_model()
    
    if model is None:
        return
    
    # Préparer les données de test
    X_test, test_ids = prepare_test_data()
    
    if X_test is None:
        return
    
    # Faire des prédictions
    submission = make_predictions(model, X_test, test_ids)
    
    if submission is None:
        return
    
    # Informations de déploiement
    serve_model_info()
    
    print("\n✅ Processus terminé!")
    print("\n📌 Fichiers créés:")
    print("   - my_submission.csv : Prédictions pour le test set")
    print("\n📌 Prochaines étapes:")
    print("   1. Vérifiez my_submission.csv")
    print("   2. (Optionnel) Servez le modèle comme API REST")
    print("   3. (Optionnel) Containerisez avec Docker")

if __name__ == "__main__":
    main()
