# 🏠 House Prices MLOps Project with MLflow

Projet MLOps complet pour la prédiction des prix immobiliers utilisant MLflow pour le tracking et la comparaison de modèles de Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10.0-orange.svg)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-green.svg)](https://scikit-learn.org/)

---

## 🎯 NOUVEAU! Documentation Complète en Français

Tous les fichiers de documentation ont été créés pour vous guider pas à pas:

| Fichier | Description | Quand l'utiliser |
|---------|-------------|------------------|
| 📌 [DEMARRER_ICI.md](DEMARRER_ICI.md) | **COMMENCEZ ICI!** Instructions de démarrage | Si vous débutez |
| 📚 [GUIDE_MLOPS_FR.md](GUIDE_MLOPS_FR.md) | Guide complet (70+ pages) | Pour tout comprendre |
| ⚡ [COMMANDES_RAPIDES.md](COMMANDES_RAPIDES.md) | Référence des commandes | Besoin d'une commande |
| 🔄 [WORKFLOW.md](WORKFLOW.md) | Diagramme du workflow MLOps | Comprendre l'architecture |
| 📋 [AIDE_MEMOIRE.md](AIDE_MEMOIRE.md) | Aide-mémoire rapide | Référence rapide |
| 📊 [RESUME_PROJET.txt](RESUME_PROJET.txt) | Résumé visuel du projet | Vue d'ensemble |

---

## 🚀 Démarrage Rapide en 3 Commandes

### Installation Express (Windows PowerShell)
```powershell
# 1. Installer toutes les dépendances automatiquement
.\install.ps1

# 2. Exécuter le pipeline complet
python src\quick_start.py

# 3. Visualiser les résultats dans MLflow UI
.\start_mlflow.ps1
```

Puis ouvrez **http://localhost:5000** dans votre navigateur!

> **💡 Conseil**: Si Python n'est pas installé, `install.ps1` vous guidera dans l'installation.

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [🚀 Démarrage Rapide](#démarrage-rapide-en-3-commandes)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modèles implémentés](#modèles-implémentés)
- [MLflow UI](#mlflow-ui)
- [Gestion des modèles](#gestion-des-modèles)

## 🎯 Aperçu

Ce projet implémente un pipeline complet de Machine Learning pour prédire les prix immobiliers (dataset Kaggle House Prices). Il utilise:

- **MLflow** pour le tracking des expérimentations
- **Docker** pour la conteneurisation
- **Scikit-learn, XGBoost, LightGBM** pour les modèles
- **Plusieurs algorithmes de régression** pour comparaison

## 📁 Structure du projet

```
house-prices-advanced-regression-techniques/
├── 📄 Dockerfile                 # Configuration Docker
├── 📄 docker-compose.yml         # Orchestration des services
├── 📄 requirements.txt           # Dépendances Python
│
├── 📊 train.csv                  # Dataset d'entraînement
├── 📊 test.csv                   # Dataset de test
├── 📊 sample_submission.csv      # Format de soumission
├── 📄 data_description.txt       # Description des données
│
├── 📂 src/                       # Code source
│   ├── data_preparation.py       # Préparation des données
│   └── train_models.py           # Pipeline d'entraînement MLflow
│
├── 📂 mlruns/                    # Stockage des expérimentations MLflow
├── 📂 models/                    # Modèles sauvegardés
└── 📂 notebooks/                 # Notebooks Jupyter (optionnel)
```

## ✅ Prérequis

- **Docker Desktop** installé et en cours d'exécution
- **Docker Compose** (inclus avec Docker Desktop)
- Au moins 4 GB de RAM disponible

## 🚀 Installation

### 1. Cloner ou naviguer vers le projet

```bash
cd c:\Users\USERµ\Desktop\MLOPS_PROJECT\house-prices-advanced-regression-techniques
```

### 2. Construire les images Docker

```bash
docker-compose build
```

Cette commande va:
- Créer une image Python avec toutes les dépendances
- Installer MLflow, scikit-learn, XGBoost, LightGBM, etc.
- Configurer l'environnement de travail

## 📖 Utilisation

### Étape 1: Démarrer les services Docker

```bash
docker-compose up -d
```

Cette commande démarre:
- **mlflow**: Service MLflow UI (accessible sur http://localhost:5000)
- **training**: Container pour exécuter les scripts d'entraînement

### Étape 2: Vérifier que les services sont actifs

```bash
docker-compose ps
```

Vous devriez voir les deux containers en cours d'exécution.

### Étape 3: Exécuter le pipeline d'entraînement

```bash
docker-compose exec training python src/train_models.py
```

Cette commande va:
1. ✅ Charger et prétraiter les données
2. ✅ Entraîner 8 modèles de régression différents
3. ✅ Logger les paramètres et métriques dans MLflow
4. ✅ Comparer les performances
5. ✅ Identifier le meilleur modèle

### Étape 4: Consulter les résultats dans MLflow UI

Ouvrez votre navigateur et allez sur:
```
http://localhost:5000
```

Vous pourrez:
- 📊 Voir toutes les expérimentations
- 📈 Comparer les modèles (RMSE, MAE, R²)
- 🔍 Analyser les paramètres de chaque modèle
- 💾 Télécharger les modèles

## 🤖 Modèles implémentés

Le pipeline entraîne et compare les modèles suivants:

| Modèle | Type | Caractéristiques |
|--------|------|------------------|
| **Linear Regression** | Linéaire | Modèle de base, rapide |
| **Ridge Regression** | Linéaire régularisé | Réduit l'overfitting (L2) |
| **Lasso Regression** | Linéaire régularisé | Sélection de features (L1) |
| **Decision Tree** | Arbre | Non-linéaire, interprétable |
| **Random Forest** | Ensemble | Robuste, réduit variance |
| **Gradient Boosting** | Ensemble | Haute performance |
| **XGBoost** | Ensemble optimisé | Très performant |
| **LightGBM** | Ensemble rapide | Efficace sur gros datasets |

### Métriques trackées

Pour chaque modèle, MLflow enregistre:
- **RMSE** (Root Mean Squared Error) - Erreur quadratique moyenne
- **MAE** (Mean Absolute Error) - Erreur absolue moyenne
- **R²** (R-squared) - Coefficient de détermination

## 🎨 MLflow UI - Guide d'utilisation

### Comparer plusieurs modèles

1. Dans MLflow UI, sélectionnez plusieurs runs
2. Cliquez sur "Compare"
3. Visualisez les graphiques de comparaison
4. Identifiez le modèle avec le meilleur R² et le plus faible RMSE

### Visualiser un modèle spécifique

1. Cliquez sur un run
2. Consultez:
   - **Parameters**: Hyperparamètres utilisés
   - **Metrics**: Performance du modèle
   - **Artifacts**: Modèle sauvegardé

## 📦 Gestion des modèles

### Enregistrer le meilleur modèle

Après avoir identifié le meilleur modèle dans MLflow UI:

```python
# Dans le container training
docker-compose exec training python

# Puis dans Python:
import mlflow
from mlflow.tracking import MlflowClient

# Initialiser le client
client = MlflowClient()

# ID du meilleur run (à récupérer depuis MLflow UI)
best_run_id = 'VOTRE_RUN_ID'
model_uri = f'runs:/{best_run_id}/model'

# Enregistrer le modèle
result = mlflow.register_model(model_uri, 'HousePricesPredictor')

# Passer en production
client.transition_model_version_stage(
    name='HousePricesPredictor',
    version=result.version,
    stage='Production'
)
```

## 🛠️ Commandes utiles

### Voir les logs en temps réel

```bash
# Logs de tous les services
docker-compose logs -f

# Logs du service MLflow uniquement
docker-compose logs -f mlflow

# Logs du service training
docker-compose logs -f training
```

### Accéder au container training

```bash
docker-compose exec training bash
```

Une fois dans le container, vous pouvez:
```bash
# Lister les fichiers
ls -la

# Voir les données
head train.csv

# Exécuter des scripts Python
python src/data_preparation.py
python src/train_models.py
```

### Re-exécuter l'entraînement

```bash
docker-compose exec training python src/train_models.py
```

### Arrêter les services

```bash
docker-compose down
```

### Supprimer tout (containers, volumes, images)

```bash
docker-compose down -v --rmi all
```

## 📊 Exemple de sortie

```
======================================================================
TRAINING COMPLETE - SUMMARY OF ALL MODELS
======================================================================

Model                     Test RMSE       Test MAE        Test R²   
----------------------------------------------------------------------
XGBoost                   $   25,432.18  $   17,234.56    0.8945
LightGBM                  $   26,123.45  $   17,891.23    0.8892
Random Forest             $   27,456.78  $   18,567.89    0.8756
Gradient Boosting         $   28,234.91  $   19,123.45    0.8634
Ridge Regression          $   32,567.89  $   22,345.67    0.8123
...

======================================================================
🏆 BEST MODEL: XGBoost
   Test RMSE: $25,432.18
   Test MAE:  $17,234.56
   Test R²:   0.8945
======================================================================

✅ All experiments logged to MLflow!
📊 View results at: http://localhost:5000
```

## 🔧 Personnalisation

### Modifier les hyperparamètres

Éditez [src/train_models.py](src/train_models.py) et modifiez les paramètres des modèles:

```python
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,      # Augmenter le nombre d'arbres
        max_depth=15,          # Profondeur maximale
        random_state=42
    ),
    # ... autres modèles
}
```

### Ajouter de nouveaux modèles

Ajoutez votre modèle dans le dictionnaire `models`:

```python
from sklearn.svm import SVR

models = {
    # ... modèles existants
    "SVR": SVR(kernel='rbf', C=1.0),
}
```

## 📚 Références

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Docker Documentation](https://docs.docker.com/)

## 🎓 Étapes suivantes

1. ✅ Expérimenter avec différents hyperparamètres
2. ✅ Effectuer une validation croisée
3. ✅ Feature engineering avancé
4. ✅ Déployer le meilleur modèle en production
5. ✅ Créer une API REST pour les prédictions
6. ✅ Intégrer un pipeline CI/CD

## 🐛 Dépannage

### Le port 5000 est déjà utilisé

Modifiez le port dans [docker-compose.yml](docker-compose.yml):
```yaml
ports:
  - "5001:5000"  # Utiliser 5001 au lieu de 5000
```

### Erreur de mémoire

Augmentez la RAM allouée à Docker Desktop dans les paramètres.

### Les volumes ne se montent pas

Sur Windows, assurez-vous que Docker Desktop a accès au disque C: dans Settings → Resources → File Sharing.

---

**Bon entraînement! 🚀**
