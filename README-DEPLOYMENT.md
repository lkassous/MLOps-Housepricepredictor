# 🏠 House Price Prediction - MLOps Project

[![CI/CD](https://github.com/YOUR_USERNAME/house-prices-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/house-prices-mlops/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)

Projet MLOps complet pour la prédiction de prix de maisons avec déploiement sur Google Cloud Platform.

## 📋 Table des Matières

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Fonctionnalités MLOps](#fonctionnalités-mlops)
- [Installation Locale](#installation-locale)
- [Déploiement Google Cloud](#déploiement-google-cloud)
- [Utilisation de l'API](#utilisation-de-lapi)
- [Structure du Projet](#structure-du-projet)
- [Coûts Estimés](#coûts-estimés)

## 🎯 Aperçu

Ce projet implémente un pipeline MLOps complet pour prédire les prix des maisons basé sur le dataset Kaggle House Prices. Il inclut:

- **ML Pipeline**: Entraînement de 3 modèles (Linear Regression, Random Forest, XGBoost)
- **MLflow**: Tracking, comparaison, et registry des modèles
- **FastAPI**: API REST pour serving des prédictions
- **Docker**: Conteneurisation multi-stage pour production
- **CI/CD**: Pipeline automatisé avec GitHub Actions
- **Cloud**: Déploiement sur Google Cloud Run

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Training   │─────▶│   MLflow     │─────▶│   Model     │
│  Pipeline   │      │   Tracking   │      │  Registry   │
└─────────────┘      └──────────────┘      └─────────────┘
                                                    │
                                                    ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│  FastAPI     │─────▶│ Production  │
│ Application │      │     API      │      │   Model     │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Google Cloud │
                    │     Run      │
                    └──────────────┘
```

## ⚙️ Fonctionnalités MLOps

### ✅ Machine Learning
- ✔️ Preprocessing automatisé (gestion valeurs manquantes, encodage)
- ✔️ Entraînement de multiples modèles
- ✔️ Hyperparameter tuning avec GridSearchCV (108 combinaisons)
- ✔️ Feature engineering avancé (30+ features)
- ✔️ Ensemble methods (Stacking)

### ✅ MLOps
- ✔️ **Experiment Tracking**: MLflow pour tous les runs
- ✔️ **Model Registry**: Gestion des versions et stages
- ✔️ **Model Serving**: API REST FastAPI
- ✔️ **Containerization**: Docker multi-stage
- ✔️ **CI/CD**: GitHub Actions
- ✔️ **Monitoring**: Health checks, metrics, logging
- ✔️ **Cloud Deployment**: Google Cloud Run

## 🚀 Installation Locale

### Prérequis
- Python 3.11+
- Docker (optionnel)
- Git

### 1. Clone du repository
```bash
git clone https://github.com/YOUR_USERNAME/house-prices-mlops.git
cd house-prices-mlops
```

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer MLflow UI
```bash
python -m mlflow ui
# Accès: http://localhost:5000
```

### 4. Entraîner les modèles
```bash
# Modèles de base
python src/train_models.py

# Hyperparameter tuning
python hyperparameter_tuning.py

# Feature engineering
python feature_engineering.py

# Modèle ensemble
python ensemble_model.py
```

### 5. Lancer l'API localement
```bash
# Sans Docker
uvicorn app:app --reload --port 8080

# Avec Docker
docker-compose -f docker-compose-prod.yml up
```

API accessible à: http://localhost:8080/docs

## ☁️ Déploiement Google Cloud

### Prérequis
1. Compte Google Cloud avec crédit de 50$
2. Google Cloud SDK installé
3. Projet GCP créé

### Étape 1: Configuration initiale

```powershell
# Installer Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authentification
gcloud auth login

# Créer un projet
gcloud projects create house-prices-mlops --name="House Prices MLOps"

# Configurer le projet
gcloud config set project house-prices-mlops

# Activer la facturation (avec votre crédit de 50$)
# https://console.cloud.google.com/billing
```

### Étape 2: Configuration des variables

```powershell
# Définir les variables d'environnement
$env:GCP_PROJECT_ID = "house-prices-mlops"
$env:GCP_REGION = "us-central1"
```

### Étape 3: Déploiement automatique

```powershell
# Exécuter le script de déploiement
.\deploy-gcp.ps1
```

Le script va:
1. ✅ Activer les APIs Google Cloud nécessaires
2. ✅ Builder l'image Docker sur Cloud Build
3. ✅ Déployer sur Cloud Run
4. ✅ Configurer autoscaling et monitoring
5. ✅ Fournir l'URL publique de l'API

### Étape 4: Configuration CI/CD (optionnel)

1. Créer un Service Account GCP:
```bash
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions"

gcloud projects add-iam-policy-binding house-prices-mlops \
    --member="serviceAccount:github-actions@house-prices-mlops.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud iam service-accounts keys create key.json \
    --iam-account=github-actions@house-prices-mlops.iam.gserviceaccount.com
```

2. Ajouter les secrets GitHub:
   - `GCP_PROJECT_ID`: house-prices-mlops
   - `GCP_SA_KEY`: contenu de key.json

3. Push sur main déclenche le déploiement automatique!

## 📡 Utilisation de l'API

### Health Check
```powershell
Invoke-WebRequest -Uri "https://YOUR-SERVICE-URL/health" | ConvertFrom-Json
```

### Prédiction unique
```powershell
$body = @{
    houses = @(
        @{
            MSSubClass = 60
            MSZoning = "RL"
            LotFrontage = 65.0
            LotArea = 8450
            # ... autres features
            OverallQual = 7
            GrLivArea = 1710
            YearBuilt = 2003
        }
    )
} | ConvertTo-Json -Depth 10

Invoke-WebRequest -Uri "https://YOUR-SERVICE-URL/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body | ConvertFrom-Json
```

### Documentation interactive
Accédez à: `https://YOUR-SERVICE-URL/docs`

## 📁 Structure du Projet

```
house-prices-advanced-regression-techniques/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # Pipeline CI/CD
├── notebooks/
│   └── house_prices_mlflow_3_models.ipynb
├── src/
│   ├── data_preparation.py
│   ├── train_models.py
│   └── register_model.py
├── tests/
│   └── test_api.py            # Tests API
├── app.py                     # FastAPI application
├── Dockerfile                 # Multi-stage Dockerfile
├── docker-compose-prod.yml    # Orchestration Docker
├── deploy-gcp.ps1             # Script déploiement PowerShell
├── deploy-gcp.sh              # Script déploiement Bash
├── requirements.txt           # Dépendances Python
├── train.csv                  # Dataset entraînement
├── test.csv                   # Dataset test
├── mlflow.db                  # Base MLflow
└── README-DEPLOYMENT.md       # Ce fichier
```

## 💰 Coûts Estimés (50$ de crédit)

### Configuration recommandée
- **Cloud Run**: 2 vCPU, 2 GB RAM
- **Cloud Build**: Build occasionnels
- **Container Registry**: Stockage images
- **Cloud Logging/Monitoring**: Logs et métriques

### Estimation mensuelle

| Service | Configuration | Coût estimé/mois |
|---------|--------------|------------------|
| Cloud Run | 2 vCPU, 2GB, min=0, max=10 | 0$ (tier gratuit) + ~5$ (requêtes) |
| Cloud Build | 10 builds/mois | Gratuit (120 min/jour) |
| Container Registry | 5 GB stockage | 0.10$ |
| Logging | 10 GB/mois | 0.50$ |
| **Total** | | **~6$/mois** |

### Avec 50$ de crédit:
- ✅ **8+ mois** de fonctionnement gratuit
- ✅ Couvre largement phase de développement et démo
- ✅ Tier gratuit Cloud Run: 2 millions requêtes/mois

### Optimisations pour réduire les coûts
1. **Min instances = 0**: Pas de coût quand pas utilisé
2. **Request timeout**: 300s max
3. **Cleanup images**: Supprimer anciennes versions
4. **Monitoring**: Alertes sur dépassement budget

```bash
# Configurer une alerte budget
gcloud billing budgets create \
    --billing-account=YOUR-BILLING-ACCOUNT \
    --display-name="MLOps Budget Alert" \
    --budget-amount=50USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90
```

## 📊 Métriques du Modèle

### Modèle en Production: HousePrices-TunedModel
- **Test RMSE**: $26,134.67
- **Test R²**: 0.9110
- **Test MAE**: $16,205.63
- **Algorithm**: XGBoost optimisé
- **Features**: 79 features engineered

## 🔒 Sécurité

- ✅ Image Docker multi-stage (réduit surface d'attaque)
- ✅ Utilisateur non-root dans container
- ✅ Variables d'environnement pour secrets
- ✅ HTTPS automatique sur Cloud Run
- ✅ Health checks et monitoring

## 📝 Licence

Ce projet est sous licence MIT.

## 👥 Auteur

Votre Nom - [GitHub](https://github.com/YOUR_USERNAME)

---

**Note**: Remplacez `YOUR_USERNAME` et `YOUR-SERVICE-URL` par vos vraies valeurs.
