# ⚡ COMMANDES RAPIDES - MLOps House Prices

## 🚀 Installation Initiale

### Étape 1: Installer Python
```powershell
# Télécharger depuis: https://www.python.org/downloads/
# ⚠️ IMPORTANT: Cochez "Add Python to PATH" pendant l'installation
```

### Étape 2: Installer toutes les dépendances
```powershell
# Méthode automatique (recommandé)
.\install.ps1

# Ou méthode manuelle
python -m pip install -r requirements.txt
```

### Étape 3: Tester l'installation
```powershell
python test_installation.py
```

---

## 🎯 Utilisation du Pipeline

### Démarrage Rapide (Tout en un)
```powershell
# Exécute: préparation → entraînement → enregistrement
python src\quick_start.py
```

### Exécution Étape par Étape
```powershell
# 1. Préparer les données
python src\data_preparation.py

# 2. Entraîner les modèles (avec MLflow tracking)
python src\train_models.py

# 3. Enregistrer le meilleur modèle
python src\register_model.py
```

---

## 📊 MLflow UI

### Démarrer MLflow UI
```powershell
# Méthode 1: Script automatique
.\start_mlflow.ps1

# Méthode 2: Commande directe
mlflow ui

# Méthode 3: Avec port personnalisé
mlflow ui --port 5001
```

### Accéder à l'interface
```
http://localhost:5000
```

### Arrêter MLflow UI
```
Ctrl + C dans le terminal
```

---

## 🔍 Commandes de Diagnostic

### Vérifier Python
```powershell
python --version
# Devrait afficher: Python 3.10.x ou supérieur
```

### Vérifier pip
```powershell
python -m pip --version
```

### Vérifier MLflow
```powershell
mlflow --version
python -c "import mlflow; print(mlflow.__version__)"
```

### Vérifier toutes les bibliothèques
```powershell
python -c "import pandas, numpy, sklearn, mlflow, xgboost, lightgbm; print('✓ Tout est OK')"
```

### Lister les packages installés
```powershell
pip list
```

---

## 📦 Gestion de l'Environnement

### Créer un environnement virtuel (recommandé)
```powershell
# Créer l'environnement
python -m venv venv

# Activer l'environnement (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activer l'environnement (Windows CMD)
.\venv\Scripts\activate.bat

# Installer les dépendances dans l'env
pip install -r requirements.txt

# Désactiver l'environnement
deactivate
```

---

## 🗂️ Gestion des Données MLflow

### Localisation des données
```
mlruns/                    # Tous les runs et expériences
mlruns/0/                  # Expérience par défaut
mlruns/<experiment_id>/    # Expérience spécifique
```

### Nettoyer les anciens runs
```powershell
# ⚠️ ATTENTION: Cela supprime TOUTES les expériences!
Remove-Item -Recurse -Force mlruns
```

### Sauvegarder les expériences
```powershell
# Copier le dossier mlruns
Copy-Item -Recurse mlruns mlruns_backup_$(Get-Date -Format 'yyyyMMdd')
```

---

## 🔧 Dépannage

### Problème: "pip n'est pas reconnu"
```powershell
# Solution: Utiliser python -m pip
python -m pip install <package>
```

### Problème: "Python n'est pas reconnu"
```powershell
# Solution 1: Redémarrer PowerShell après installation
# Solution 2: Ajouter Python au PATH manuellement
# Solution 3: Réinstaller Python en cochant "Add to PATH"
```

### Problème: MLflow UI ne démarre pas
```powershell
# Vérifier si le port 5000 est utilisé
netstat -ano | findstr :5000

# Utiliser un autre port
mlflow ui --port 5001
```

### Problème: Erreur d'import
```powershell
# Réinstaller les dépendances
pip install --upgrade --force-reinstall -r requirements.txt
```

### Problème: Dataset non trouvé
```powershell
# Vérifier le dossier actuel
Get-Location

# Naviguer vers le bon dossier
cd c:\Users\USERµ\Desktop\MLOPS_PROJECT\house-prices-advanced-regression-techniques

# Vérifier que train.csv existe
Test-Path train.csv
```

---

## 🐳 Utilisation avec Docker (Optionnel)

### Construire l'image
```powershell
docker-compose build
```

### Démarrer les conteneurs
```powershell
docker-compose up
```

### Arrêter les conteneurs
```powershell
docker-compose down
```

---

## 📈 Expérimentation et Optimisation

### Modifier les paramètres des modèles
Éditez `src\train_models.py` et ajustez les hyperparamètres:
```python
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,  # Changer de 100 à 200
        max_depth=15,      # Augmenter la profondeur
        random_state=42
    ),
}
```

### Ajouter un nouveau modèle
Dans `src\train_models.py`, ajoutez au dictionnaire `models`:
```python
from sklearn.svm import SVR

models = {
    # ... modèles existants ...
    "Support Vector Regression": SVR(kernel='rbf', C=1.0),
}
```

### Créer une nouvelle expérience
```python
import mlflow
mlflow.set_experiment("Mon-Nouvelle-Experience")
```

---

## 📊 Analyse des Résultats

### Via MLflow UI
1. Ouvrir http://localhost:5000
2. Sélectionner l'expérience "House-Prices-Regression"
3. Trier par métrique (RMSE, R², MAE)
4. Comparer plusieurs runs

### Via Python
```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("House-Prices-Regression")

# Obtenir tous les runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_r2 DESC"]
)

# Afficher le meilleur
best_run = runs[0]
print(f"Meilleur modèle: {best_run.data.params['model_name']}")
print(f"R²: {best_run.data.metrics['test_r2']:.4f}")
```

---

## 🎓 Ressources Utiles

### Documentation
- MLflow: https://mlflow.org/docs/latest/
- Scikit-learn: https://scikit-learn.org/stable/
- Pandas: https://pandas.pydata.org/docs/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/

### Fichiers du Projet
- `GUIDE_MLOPS_FR.md` - Guide complet en français
- `README.md` - README principal
- `requirements.txt` - Liste des dépendances
- `test_installation.py` - Script de test

---

## 💡 Conseils Pro

### Performance
- Utilisez un environnement virtuel pour isoler les dépendances
- Nettoyez régulièrement les anciens runs MLflow
- Sauvegardez vos meilleurs modèles

### Organisation
- Nommez vos expériences de manière descriptive
- Ajoutez des tags aux runs importants dans MLflow UI
- Documentez vos découvertes dans les notes MLflow

### Workflow
1. Commencez avec des modèles simples
2. Comparez les résultats dans MLflow UI
3. Optimisez les hyperparamètres des meilleurs modèles
4. Enregistrez le modèle final dans le Model Registry

---

**Pour plus de détails, consultez [GUIDE_MLOPS_FR.md](GUIDE_MLOPS_FR.md)**
