# 🔄 MLOps Pipeline - Documentation

## Vue d'ensemble

Ce pipeline automatise le flux complet de machine learning pour la prédiction des prix immobiliers :

```
CSV Data Input → Validation → Prétraitement → Entraînement (3 modèles) → Évaluation → Sélection → MLflow Registry → Production
```

---

## 📊 Composants du Pipeline

### 1. **pipeline.py** (407 lignes)
Script principal qui orchestrate tout le pipeline MLOps.

#### Fonctionnalités :
- ✅ **Chargement & Validation** : Charge le CSV, vérifie les données
- ✅ **Prétraitement** : Gère 43 colonnes catégorielles, imputation médiane/mode
- ✅ **Entraînement** : 3 modèles en parallèle
  - Linear Regression
  - Random Forest (100 estimators, max_depth=10)
  - XGBoost (300 estimators, max_depth=3, learning_rate=0.1)
- ✅ **Évaluation** : Calcule RMSE, MAE, R², cross-validation 5-fold
- ✅ **Sélection automatique** : Sélectionne le meilleur modèle (basé sur Test R²)
- ✅ **MLflow Logging** : Logue tous les modèles, métriques, paramètres
- ✅ **Promotion Production** : Auto-promeut le meilleur en @production
- ✅ **Rapports** : Génère pipeline_report.json

#### Utilisation locale :
```bash
python pipeline.py --data-path train.csv --output-path ./mlruns
```

#### Sortie attendue :
```
========================================================================
STARTING MLOPS PIPELINE
========================================================================

✅ Data loaded: 1460 rows, 81 columns
✅ Data validation complete
✅ Duplicates removed: 1460 rows remaining
✅ Missing values handled
✅ Categorical variables encoded: 43 encoders
✅ Data split: Train=1168, Test=292
   Features: 79

========================================================================
TRAINING ALL MODELS
========================================================================

Training LinearRegression...
✅ LinearRegression trained
   Test RMSE: $35,312.14
   Test R²: 0.8374

Training RandomForest...
✅ RandomForest trained
   Test RMSE: $28,762.74
   Test R²: 0.8921

Training XGBoost...
✅ XGBoost trained
   Test RMSE: $25,089.72
   Test R²: 0.9179

========================================================================
MODEL COMPARISON
========================================================================

Model Performance Ranking (sorted by Test R²):
           test_rmse  test_mae  test_r2
XGBoost      25089.72  15953.63  0.9179
RandomForest 28762.74  17804.52  0.8921
LinearRegression 35312.14  21580.19  0.8374

🏆 BEST MODEL: XGBoost
   Test RMSE: $25,089.72
   Test R²: 0.9179

========================================================================
LOGGING TO MLFLOW
========================================================================

✅ Logged LinearRegression to MLflow
✅ Logged RandomForest to MLflow
✅ Logged XGBoost to MLflow

========================================================================
PROMOTING BEST MODEL TO PRODUCTION
========================================================================

✅ Model registered: HousePricesPredictor
   Version: 3
   Run ID: abc123def456

✅ Alias 'production' set to version 3

========================================================================
✅ PIPELINE COMPLETED SUCCESSFULLY
========================================================================

📊 Summary:
   Best Model: XGBoost
   MLflow Experiment: House-Prices-Production-Pipeline
   Model Registry: HousePricesPredictor v3

🌐 Access MLflow UI: mlflow ui --host 0.0.0.0 --port 5000
```

---

### 2. **.github/workflows/mlops-pipeline.yml** (170 lignes)
Workflow GitHub Actions qui automatise le pipeline en CI/CD.

#### Déclencheurs :
- ✅ Push de `train.csv` vers la branche principale
- ✅ Push de `pipeline.py` ou `requirements.txt`
- ✅ Workflow manuel (workflow_dispatch)

#### Étapes du workflow :
```
1. Checkout code
2. Setup Python 3.11
3. Install dependencies (pip)
4. Run MLOps Pipeline (python pipeline.py)
5. Generate artifacts (MLflow, reports)
6. Upload MLflow artifacts → GitHub
7. Build Docker image
8. Push to Google Container Registry
9. Deploy to Google Cloud Run
10. Notify on success/failure
```

#### Configuration requise :
```yaml
GitHub Repository Secrets:
  - GCP_PROJECT_ID: "votre-gcp-project-id"
  - GCP_SA_KEY: "contenu du fichier JSON de la service account"
```

#### Commandes pour configurer :
```bash
# Générer une service account GCP
# 1. Aller à https://console.cloud.google.com
# 2. Créer une service account avec les permissions Cloud Run Admin
# 3. Télécharger la clé JSON
# 4. Ajouter comme secret GitHub: GCP_SA_KEY

# Trouver ton Project ID
gcloud config get-value project

# Ajouter comme secret GitHub: GCP_PROJECT_ID
```

---

### 3. **config.yaml** (Configuration centralisée)
Fichier YAML qui centralise toute la configuration du pipeline.

```yaml
# Data Configuration
data:
  target_column: "SalePrice"
  test_size: 0.2
  random_state: 42

# Model Hyperparameters
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  xgboost:
    n_estimators: 300
    max_depth: 3
    learning_rate: 0.1
    random_state: 42

# MLflow Configuration
mlflow:
  experiment_name: "House-Prices-Production-Pipeline"
  model_registry_name: "HousePricesPredictor"
  backend_store_uri: "./mlruns"
  production_alias: "production"
  min_r2_threshold: 0.80
```

---

### 4. **data_schema.json** (Validation des données)
Schéma qui décrit la structure attendue des données.

```json
{
  "schema": {
    "target_variable": "SalePrice",
    "features": {
      "numeric": {
        "count": 36,
        "examples": ["LotArea", "OverallQual", "YearBuilt", ...]
      },
      "categorical": {
        "count": 43,
        "examples": ["MSZoning", "Neighborhood", "BldgType", ...]
      }
    },
    "data_size": {
      "samples": 1460,
      "features": 79,
      "train_test_split": "80/20"
    }
  }
}
```

---

## 🔄 Flux d'exécution détaillé

### Phase 1 : Chargement et Validation
```python
df = load_data(data_path)              # Charge CSV
report = validate_data(df)             # Vérifie intégrité des données
```

**Vérifications** :
- ✅ Fichier existe
- ✅ Colonne cible 'SalePrice' présente
- ✅ Nombre de lignes ≥ 100
- ✅ Nombre de colonnes ≥ 50

---

### Phase 2 : Prétraitement
```python
X_train, X_test, y_train, y_test, label_encoders, feature_names = preprocess_data(df)
```

**Étapes** :
1. **Suppression des doublons** : Élimine les lignes dupliquées
2. **Gestion valeurs manquantes** :
   - Numériques : remplissage médiane
   - Catégorielles : remplissage mode
3. **Encodage catégories** : LabelEncoder pour 43 colonnes
4. **Séparation train/test** : 80/20 split, 1168 train / 292 test
5. **Feature engineering** : 79 features finales

---

### Phase 3 : Entraînement
```python
trained_models, all_metrics = train_models(X_train, X_test, y_train, y_test)
```

**Modèles entraînés en parallèle** :

| Modèle | Hyperparam | Test RMSE | Test R² |
|--------|-----------|----------|---------|
| Linear Regression | - | $35,312 | 0.8374 |
| Random Forest | n=100, d=10 | $28,762 | 0.8921 |
| XGBoost | n=300, d=3, lr=0.1 | $25,089 | 0.9179 ✅ |

**Métriques calculées** :
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient de détermination)
- Cross-Validation 5-Fold

---

### Phase 4 : Évaluation et Sélection
```python
best_model, comparison_df = compare_models(all_metrics)
```

**Critère de sélection** : Test R² (plus élevé = meilleur)

Résultat : **XGBoost** (R²=0.9179)

---

### Phase 5 : MLflow Logging
```python
run_ids, experiment_id = log_models_to_mlflow(trained_models, all_metrics, best_model)
```

**Loggé par modèle** :
- ✅ Modèle complet (pickle)
- ✅ Tous les paramètres
- ✅ Toutes les métriques
- ✅ Artifacts (requirements.txt, etc.)

**Accessible via** :
```bash
mlflow ui --host 0.0.0.0 --port 5000
# Ouvre http://localhost:5000
```

---

### Phase 6 : Promotion Production
```python
version = promote_best_model_to_production(best_model, run_ids)
```

**Action** :
- ✅ Enregistre le modèle au Registry MLflow
- ✅ Ajoute l'alias `@production`
- ✅ Rend disponible pour serving

---

## 📈 Monitoring et Rapports

### MLflow UI
Affiche en temps réel :
- 📊 Comparaison des runs
- 📈 Graphiques de métriques
- 🔍 Détails des artefacts
- 🎯 Historique complet

### Rapports JSON
```bash
pipeline_output/pipeline_report.json
```

Contient :
```json
{
  "timestamp": "2026-01-04T12:30:45.123456",
  "data_quality": {
    "total_rows": 1460,
    "total_columns": 81,
    "duplicates": 0
  },
  "best_model": "XGBoost",
  "model_metrics": {
    "XGBoost": {
      "test_rmse": 25089.72,
      "test_r2": 0.9179,
      "cv_mean": 0.9156,
      "cv_std": 0.0089
    }
  },
  "pipeline_status": "SUCCESS"
}
```

---

## 🚀 Utilisation en Production

### Déploiement sur Cloud Run
```bash
# GitHub Actions fait cela automatiquement !
# Sinon, manuellement :

gcloud run deploy house-prices-predictor \
  --image=gcr.io/PROJECT_ID/house-prices-api:latest \
  --region=us-central1 \
  --allow-unauthenticated \
  --port=8080
```

### Prédictions via API
```bash
curl -X POST https://house-prices-predictor-xxx.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MSZoning": "RL",
    "LotArea": 8450,
    "OverallQual": 7,
    ...
  }'

# Réponse
{
  "prediction": 180500.25,
  "model_name": "HousePricesPredictor",
  "model_version": "@production",
  "prediction_time_ms": 48.3
}
```

---

## 📊 Notebook Amélioré

Le notebook [house_prices_mlflow_3_models.ipynb](notebooks/house_prices_mlflow_3_models.ipynb) inclut **15 cellules** :

**Cellules 1-10** : MLflow tracking + 3 modèles
**Cellules 11-15** : Visualisations avancées

1. **Feature Importance** : Top 10 features (Random Forest + XGBoost)
2. **Residual Analysis** : Erreurs de prédiction par modèle
3. **Learning Curves** : Surfit/underfit detection
4. **Actual vs Predicted** : Scatter plots de précision
5. **Cross-Validation** : 5-fold CV analysis

---

## ✅ Checklist d'implémentation

### Local Testing
- [ ] `python pipeline.py --data-path train.csv`
- [ ] Vérifier pipeline_report.json généré
- [ ] `mlflow ui` et consulter l'interface

### GitHub Setup
- [ ] Push code vers GitHub
- [ ] Ajouter secrets : `GCP_PROJECT_ID`, `GCP_SA_KEY`
- [ ] Vérifier GitHub Actions en passant

### GCP Deployment
- [ ] Créer GCP Project
- [ ] Créer Service Account avec Cloud Run Admin
- [ ] Télécharger clé JSON
- [ ] Configurer GitHub secrets

### Automated Workflow
- [ ] Pousser nouveau train.csv
- [ ] Vérifier workflow GitHub Actions
- [ ] Vérifier déploiement Cloud Run
- [ ] Tester API en production

---

## 🐛 Troubleshooting

### Le pipeline échoue localement
```bash
# Vérifier les dépendances
pip install -r requirements.txt

# Vérifier le CSV
python -c "import pandas as pd; df = pd.read_csv('train.csv'); print(df.shape)"

# Lancer avec verbose
python pipeline.py --data-path train.csv 2>&1 | tail -50
```

### MLflow UI ne démarre pas
```bash
# Tuer les processus existants
lsof -i :5000        # macOS/Linux
Get-Process -Port 5000  # Windows

# Lancer depuis le bon dossier
cd .../house-prices-advanced-regression-techniques
mlflow ui --host 0.0.0.0 --port 5000
```

### GitHub Actions échoue
```bash
# Vérifier les logs
# 1. Aller à Actions dans GitHub
# 2. Cliquer sur le workflow
# 3. Voir les logs de chaque étape
```

---

## 📞 Support

- 📧 Email : lkassous17@gmail.com
- 🔗 GitHub : https://github.com/lkassous/MLOps-Housepricepredictor
- 📚 Docs : Voir [GUIDE_MLOPS_FR.md](GUIDE_MLOPS_FR.md)

---

**Créé avec ❤️ pour la communauté MLOps**
