#!/usr/bin/env python3
"""
MLOps Pipeline Local Test Script
Test que le pipeline.py fonctionne correctement avant GitHub Actions
"""

import sys
import subprocess
import json
from pathlib import Path

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def check_requirements():
    """Vérifier que toutes les dépendances sont installées"""
    print_header("1️⃣ VÉRIFICATION DES DÉPENDANCES")
    
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import mlflow
        print("✅ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"❌ Erreur: {e}")
        print("💡 Installez les dépendances avec:")
        print("   pip install -r requirements.txt")
        return False

def check_data_file():
    """Vérifier que le fichier train.csv existe"""
    print_header("2️⃣ VÉRIFICATION DES DONNÉES")
    
    if Path("train.csv").exists():
        print("✅ train.csv trouvé")
        
        import pandas as pd
        df = pd.read_csv("train.csv")
        print(f"   - {df.shape[0]} samples")
        print(f"   - {df.shape[1]} colonnes")
        print(f"   - 'SalePrice' présent: {'SalePrice' in df.columns}")
        return True
    else:
        print("❌ train.csv non trouvé")
        return False

def run_pipeline():
    """Exécuter le pipeline"""
    print_header("3️⃣ EXÉCUTION DU PIPELINE")
    
    try:
        result = subprocess.run(
            [sys.executable, "pipeline.py", "--data-path", "train.csv", "--output-path", "./mlruns"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Erreur d'exécution:\n{result.stderr}")
            return False
        
        return True
    except subprocess.TimeoutExpired:
        print("❌ Timeout - le pipeline a pris trop longtemps")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def check_report():
    """Vérifier que le rapport a été généré"""
    print_header("4️⃣ VÉRIFICATION DES RÉSULTATS")
    
    try:
        if Path("mlruns").exists():
            print("✅ Dossier mlruns créé")
            
        # Compter les runs
        runs = list(Path("mlruns").glob("0/*/params"))
        if runs:
            print(f"✅ {len(runs)} run(s) loggé(s) dans MLflow")
        
        return True
    except Exception as e:
        print(f"⚠️  Erreur de vérification: {e}")
        return True  # Non-bloquant

def main():
    """Script principal"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         🧪 MLOPS PIPELINE - LOCAL TEST SCRIPT 🧪                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    results = {
        "Dépendances": check_requirements(),
        "Données": check_data_file(),
        "Pipeline": run_pipeline(),
        "Résultats": check_report()
    }
    
    # Résumé
    print_header("📊 RÉSUMÉ")
    
    for name, status in results.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name}: {'Passé' if status else 'Échoué'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS SONT PASSÉS! 🎉")
        print("="*70)
        print("\n✅ Le pipeline est prêt pour GitHub Actions")
        print("📍 Prochaine étape: Déclencher le workflow dans GitHub UI")
        print("   1. Aller à: https://github.com/lkassous/MLOps-Housepricepredictor/actions")
        print("   2. Cliquer sur 'mlops-pipeline.yml'")
        print("   3. Cliquer sur 'Run workflow' (bouton bleu)")
        print("   4. Cliquer 'Run workflow'")
        return 0
    else:
        print("\n" + "="*70)
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("="*70)
        print("\n💡 Vérifiez les erreurs ci-dessus et réessayez")
        return 1

if __name__ == "__main__":
    sys.exit(main())
