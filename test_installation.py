"""
Script de Test Rapide - Vérification de l'Installation MLOps
Teste que toutes les bibliothèques et le dataset sont prêts
"""

import sys

print("=" * 70)
print("TEST DE L'INSTALLATION MLOPS")
print("=" * 70)
print()

# Test 1: Bibliothèques essentielles
print("1. Test des bibliothèques essentielles...")
print("-" * 70)

libraries = [
    ("MLflow", "mlflow"),
    ("Pandas", "pandas"),
    ("NumPy", "numpy"),
    ("Scikit-learn", "sklearn"),
    ("XGBoost", "xgboost"),
    ("LightGBM", "lightgbm"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
]

all_ok = True
for lib_name, lib_import in libraries:
    try:
        module = __import__(lib_import)
        version = getattr(module, '__version__', 'N/A')
        print(f"  ✓ {lib_name:<20} version {version}")
    except ImportError:
        print(f"  ✗ {lib_name:<20} NON INSTALLÉ!")
        all_ok = False

print()

if not all_ok:
    print("❌ Certaines bibliothèques sont manquantes!")
    print("   Exécutez: python -m pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Dataset
print("2. Test du dataset...")
print("-" * 70)

import os
import pandas as pd

if os.path.exists('train.csv'):
    print("  ✓ train.csv trouvé")
    try:
        df = pd.read_csv('train.csv')
        print(f"  ✓ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        if 'SalePrice' in df.columns:
            print(f"  ✓ Colonne cible 'SalePrice' présente")
            print(f"  ℹ Prix moyen: ${df['SalePrice'].mean():,.2f}")
            print(f"  ℹ Prix min: ${df['SalePrice'].min():,.2f}")
            print(f"  ℹ Prix max: ${df['SalePrice'].max():,.2f}")
        else:
            print("  ✗ Colonne 'SalePrice' manquante!")
            all_ok = False
    except Exception as e:
        print(f"  ✗ Erreur lors du chargement: {e}")
        all_ok = False
else:
    print("  ✗ train.csv non trouvé!")
    print("     Assurez-vous d'être dans le bon dossier")
    all_ok = False

if os.path.exists('test.csv'):
    print("  ✓ test.csv trouvé")
else:
    print("  ⚠ test.csv non trouvé (optionnel)")

print()

# Test 3: Structure du projet
print("3. Test de la structure du projet...")
print("-" * 70)

folders = ['src', 'mlruns', 'models']
for folder in folders:
    if os.path.exists(folder):
        print(f"  ✓ Dossier '{folder}/' présent")
    else:
        print(f"  ⚠ Dossier '{folder}/' absent (sera créé automatiquement)")

print()

# Test 4: Scripts Python
print("4. Test des scripts Python...")
print("-" * 70)

scripts = [
    'src/data_preparation.py',
    'src/train_models.py',
    'src/register_model.py',
    'src/quick_start.py'
]

for script in scripts:
    if os.path.exists(script):
        print(f"  ✓ {script}")
    else:
        print(f"  ✗ {script} manquant!")
        all_ok = False

print()

# Test 5: MLflow
print("5. Test de MLflow...")
print("-" * 70)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("  ✓ MLflow importé avec succès")
    print(f"  ✓ Version: {mlflow.__version__}")
    
    # Tester la création d'un client MLflow
    client = MlflowClient()
    print("  ✓ Client MLflow initialisé")
    
except Exception as e:
    print(f"  ✗ Erreur MLflow: {e}")
    all_ok = False

print()

# Test 6: Test rapide de préparation des données
print("6. Test rapide de préparation des données...")
print("-" * 70)

try:
    from src.data_preparation import load_data, preprocess_data
    
    print("  ✓ Modules de préparation importés")
    
    # Charger les données
    df = load_data('train.csv')
    print(f"  ✓ Données chargées: {df.shape}")
    
    # Prétraiter
    X, y, encoders = preprocess_data(df)
    print(f"  ✓ Prétraitement réussi: {X.shape[1]} features")
    print(f"  ✓ {len(encoders)} encodeurs créés")
    
except Exception as e:
    print(f"  ✗ Erreur lors du test: {e}")
    all_ok = False

print()
print("=" * 70)

if all_ok:
    print("✅ TOUS LES TESTS SONT PASSÉS!")
    print("=" * 70)
    print()
    print("🎉 Votre environnement MLOps est prêt!")
    print()
    print("PROCHAINES ÉTAPES:")
    print()
    print("1. Entraîner les modèles:")
    print("   python src/train_models.py")
    print()
    print("2. Ou utiliser le démarrage rapide:")
    print("   python src/quick_start.py")
    print()
    print("3. Visualiser dans MLflow UI:")
    print("   mlflow ui")
    print("   Puis ouvrir: http://localhost:5000")
    print()
    print("📚 Guide complet: Consultez GUIDE_MLOPS_FR.md")
    print()
else:
    print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 70)
    print()
    print("Veuillez corriger les erreurs ci-dessus avant de continuer.")
    print()
    print("AIDE:")
    print("- Installer les dépendances: python -m pip install -r requirements.txt")
    print("- Vérifier que vous êtes dans le bon dossier")
    print("- Consulter GUIDE_MLOPS_FR.md pour plus d'aide")
    print()
    sys.exit(1)
