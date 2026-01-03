# 🏠 Guide MLOps - Prédiction des Prix de Maisons avec MLflow

## 📋 Table des Matières
1. [Prérequis](#prérequis)
2. [Installation de Python](#installation-de-python)
3. [Installation des Dépendances](#installation-des-dépendances)
4. [Structure du Projet](#structure-du-projet)
5. [Utilisation du Pipeline MLflow](#utilisation-du-pipeline-mlflow)
6. [Comparaison des Modèles](#comparaison-des-modèles)
7. [Enregistrement des Modèles](#enregistrement-des-modèles)
8. [Déploiement](#déploiement)

---

## 🎯 Prérequis

### Installation de Python
1. **Télécharger Python 3.10+**
   - Allez sur [python.org](https://www.python.org/downloads/)
   - Téléchargez Python 3.10 ou plus récent pour Windows
   - ⚠️ **IMPORTANT**: Cochez "Add Python to PATH" pendant l'installation

2. **Vérifier l'installation**
   ```powershell
   python --version
   # Devrait afficher: Python 3.10.x ou plus
   ```

---

## 📦 Installation des Dépendances

### Méthode 1: Installation depuis requirements.txt (Recommandé)
```powershell
# Naviguer vers le dossier du projet
cd c:\Users\USERµ\Desktop\MLOPS_PROJECT\house-prices-advanced-regression-techniques

# Installer toutes les dépendances
python -m pip install -r requirements.txt
```

### Méthode 2: Installation manuelle
```powershell
# MLflow et outils de tracking
python -m pip install mlflow==2.10.0

# Bibliothèques ML essentielles
python -m pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.2

# Modèles avancés
python -m pip install xgboost==2.0.3 lightgbm==4.1.0

# Visualisation
python -m pip install matplotlib==3.7.3 seaborn==0.13.0

# Utilitaires
python -m pip install joblib==1.3.2 scipy==1.11.4
```

### Vérification de l'installation
```powershell
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

---

## 📁 Structure du Projet

```
house-prices-advanced-regression-techniques/
│
├── 📄 train.csv                    # Dataset d'entraînement (1460 maisons)
├── 📄 test.csv                     # Dataset de test
├── 📄 requirements.txt             # Dépendances Python
├── 📄 docker-compose.yml           # Configuration Docker
├── 📄 Dockerfile                   # Image Docker
│
├── 📂 src/                         # Code source
│   ├── data_preparation.py        # Préparation des données
│   ├── train_models.py            # Entraînement avec MLflow
│   ├── register_model.py          # Enregistrement du meilleur modèle
│   └── quick_start.py             # Script de démarrage rapide
│
├── 📂 mlruns/                      # Stockage MLflow (généré automatiquement)
├── 📂 models/                      # Modèles sauvegardés
└── 📂 notebooks/                   # Notebooks Jupyter (analyse exploratoire)
```

---

## 🚀 Utilisation du Pipeline MLflow

### Étape 1: Préparation des Données

Le script `data_preparation.py` charge et prépare les données:
- ✅ Chargement du dataset (81 features)
- ✅ Gestion des valeurs manquantes
- ✅ Encodage des variables catégorielles
- ✅ Séparation train/test (80/20)

```powershell
# Exécuter la préparation des données
python src/data_preparation.py
```

**Sortie attendue:**
```
Dataset loaded successfully with shape: (1460, 81)
Numeric features: 36
Categorical features: 43
Preprocessing complete. Final shape: (1460, 79)
Training set size: 1168
Test set size: 292
```

---

### Étape 2: Entraînement des Modèles avec MLflow

Le script `train_models.py` entraîne **8 modèles différents** et log tout dans MLflow:

#### 🤖 Modèles Entraînés:
1. **Linear Regression** - Régression linéaire simple
2. **Ridge Regression** - Régression avec régularisation L2
3. **Lasso Regression** - Régression avec régularisation L1
4. **Decision Tree** - Arbre de décision
5. **Random Forest** - Forêt aléatoire (100 arbres)
6. **Gradient Boosting** - Boosting de gradients
7. **XGBoost** - Extreme Gradient Boosting
8. **LightGBM** - Light Gradient Boosting Machine

#### 📊 Métriques Suivies:
- **RMSE** (Root Mean Squared Error) - Erreur quadratique moyenne
- **MAE** (Mean Absolute Error) - Erreur absolue moyenne
- **R²** (R-squared) - Coefficient de détermination

```powershell
# Entraîner tous les modèles
python src/train_models.py
```

**Ce qui est logué dans MLflow:**
- ✅ Nom du modèle et tous ses paramètres
- ✅ Métriques de performance (train et test)
- ✅ Le modèle entraîné lui-même
- ✅ Durée d'entraînement
- ✅ Dataset utilisé

---

### Étape 3: Visualiser les Résultats avec MLflow UI

#### Démarrer le serveur MLflow:
```powershell
# Dans le dossier du projet
mlflow ui
```

#### Accéder à l'interface:
1. Ouvrez votre navigateur web
2. Allez à: **http://localhost:5000** (ou http://127.0.0.1:5000)

#### 🎨 Interface MLflow UI - Fonctionnalités:

**Page d'accueil:**
- 📋 Liste de tous les runs (expériences)
- 🔍 Filtrage et recherche par modèle
- 📊 Tri par métriques (RMSE, R², MAE)

**Comparaison de modèles:**
1. Cochez les modèles à comparer
2. Cliquez sur "Compare"
3. Visualisez:
   - Graphiques de métriques
   - Différences de paramètres
   - Courbes de performance

**Détails d'un Run:**
- Tous les paramètres du modèle
- Toutes les métriques
- Artifacts (modèle sauvegardé)
- Code source utilisé

---

### Étape 4: Enregistrement du Meilleur Modèle

#### Méthode Automatique:
```powershell
# Le script trouve automatiquement le meilleur modèle (par R²)
python src/register_model.py
```

#### Méthode Manuelle (via MLflow UI):
1. Dans MLflow UI, trouvez le meilleur run
2. Cliquez sur le run
3. Dans la section "Artifacts", cliquez sur "Register Model"
4. Nom du modèle: `HousePricesPredictor`
5. Choisissez la version et le stage (Staging/Production)

#### 🏷️ Stages du Modèle:
- **None** - Modèle non déployé
- **Staging** - En phase de test
- **Production** - En production
- **Archived** - Archivé

---

## ⚡ Démarrage Rapide (Quick Start)

Pour exécuter tout le pipeline automatiquement:

```powershell
# Exécute: préparation des données → entraînement → enregistrement
python src/quick_start.py
```

Ensuite, démarrez MLflow UI:
```powershell
mlflow ui
```

---

## 📊 Exemple de Résultats Attendus

### Sortie du Training:
```
==================================================
Training: Random Forest
==================================================

Training Results:
  RMSE: $15,234.56
  MAE:  $10,123.45
  R²:   0.9567

Test Results:
  RMSE: $28,456.78
  MAE:  $18,234.56
  R²:   0.8734
```

### Tableau Récapitulatif:
```
Model                     Test RMSE       Test MAE        Test R²
----------------------------------------------------------------------
XGBoost                   $25,123.45      $16,789.12      0.8956
LightGBM                  $26,234.56      $17,234.56      0.8912
Random Forest             $28,456.78      $18,234.56      0.8734
Gradient Boosting         $29,123.45      $19,456.78      0.8678
Ridge Regression          $32,456.78      $21,234.56      0.8456
Lasso Regression          $33,234.56      $22,123.45      0.8398
Linear Regression         $34,567.89      $23,456.78      0.8234
Decision Tree             $38,123.45      $25,678.90      0.7956
```

🏆 **Meilleur Modèle: XGBoost** (R² = 0.8956)

---

## 🔄 Workflow Complet MLOps

### Diagramme de Flux:
```
1. Préparation des Données (data_preparation.py)
   ↓
2. Entraînement des Modèles (train_models.py)
   ↓
3. Tracking avec MLflow (automatique)
   ↓
4. Visualisation (MLflow UI)
   ↓
5. Sélection du Meilleur Modèle
   ↓
6. Enregistrement (register_model.py)
   ↓
7. Déploiement en Production
```

---

## 🛠️ Commandes Utiles

### Gestion de MLflow:
```powershell
# Démarrer MLflow UI
mlflow ui

# Démarrer sur un port différent
mlflow ui --port 8080

# Voir la version de MLflow
mlflow --version

# Nettoyer les anciens runs (attention!)
# Supprimer le dossier mlruns pour recommencer
Remove-Item -Recurse -Force mlruns
```

### Développement:
```powershell
# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement (Windows)
.\venv\Scripts\Activate.ps1

# Installer les dépendances dans l'env virtuel
pip install -r requirements.txt
```

---

## 📈 Optimisation et Hyperparamètres

### Modifier les Paramètres des Modèles:

Éditez `train_models.py` pour ajuster les hyperparamètres:

```python
models = {
    # Augmenter le nombre d'arbres dans Random Forest
    "Random Forest": RandomForestRegressor(
        n_estimators=200,      # 100 → 200 arbres
        max_depth=15,          # 10 → 15 profondeur max
        random_state=42
    ),
    
    # Ajuster XGBoost
    "XGBoost": XGBRegressor(
        n_estimators=150,      # Plus d'itérations
        max_depth=7,           # Profondeur augmentée
        learning_rate=0.05,    # Learning rate plus faible
        random_state=42
    ),
}
```

Puis réentraînez:
```powershell
python src/train_models.py
```

MLflow créera de nouveaux runs que vous pourrez comparer avec les anciens!

---

## 🐳 Utilisation avec Docker (Optionnel)

### Construire l'image:
```powershell
docker-compose build
```

### Lancer le conteneur:
```powershell
docker-compose up
```

---

## 🔍 Troubleshooting

### Problème: MLflow UI ne démarre pas
```powershell
# Vérifier si le port 5000 est déjà utilisé
netstat -ano | findstr :5000

# Utiliser un autre port
mlflow ui --port 5001
```

### Problème: Import Error
```powershell
# Réinstaller les dépendances
pip install --upgrade -r requirements.txt
```

### Problème: Données non trouvées
```powershell
# S'assurer d'être dans le bon dossier
cd c:\Users\USERµ\Desktop\MLOPS_PROJECT\house-prices-advanced-regression-techniques

# Vérifier que train.csv existe
dir train.csv
```

---

## 📚 Ressources Supplémentaires

### Documentation:
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **Scikit-learn**: https://scikit-learn.org/stable/
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/

### Tutoriels:
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
- MLflow Models: https://mlflow.org/docs/latest/models.html
- Model Registry: https://mlflow.org/docs/latest/model-registry.html

---

## 🎯 Prochaines Étapes

1. ✅ **Expérimenter avec d'autres modèles**
   - Essayez CatBoost, Support Vector Regression
   - Combinez plusieurs modèles (Stacking/Blending)

2. ✅ **Optimisation des Hyperparamètres**
   - Utilisez GridSearchCV ou RandomizedSearchCV
   - Intégrez Optuna pour l'optimisation Bayésienne

3. ✅ **Feature Engineering**
   - Créez de nouvelles features
   - Sélectionnez les features importantes
   - Analysez les corrélations

4. ✅ **Déploiement**
   - Créez une API REST avec FastAPI
   - Déployez sur le cloud (Azure, AWS, GCP)
   - Configurez un pipeline CI/CD

5. ✅ **Monitoring en Production**
   - Surveillez les performances du modèle
   - Détectez le drift des données
   - Automatisez le réentraînement

---

## 👥 Support

Pour toute question ou problème:
- Consultez la documentation MLflow
- Vérifiez les logs dans mlruns/
- Examinez les messages d'erreur dans le terminal

---

## 📝 Licence

Ce projet utilise le dataset "House Prices - Advanced Regression Techniques" de Kaggle.

---

**Bon apprentissage automatique! 🚀**
