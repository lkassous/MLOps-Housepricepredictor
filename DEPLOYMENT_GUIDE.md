# 🚀 Guide de Déploiement Google Cloud Run

## ✅ État Actuel
- Image Docker: `house-prices-api:latest` (testée et fonctionnelle)
- API locale: http://localhost:8080 ✅ WORKING
- Prédiction testée: $197,460 pour une maison (48ms)
- Crédit GCP: $50 disponible

## 📋 Prérequis

### 1. Installer Google Cloud SDK
```powershell
# Télécharger depuis: https://cloud.google.com/sdk/docs/install-sdk#windows
# Ou avec Chocolatey:
choco install gcloudsdk

# Vérifier l'installation
gcloud --version
```

### 2. Authentification
```powershell
# Se connecter à Google Cloud
gcloud auth login

# Lister les projets disponibles
gcloud projects list

# Ou créer un nouveau projet
gcloud projects create house-prices-mlops --name="House Prices MLOps"
```

### 3. Configurer le projet
```powershell
# Définir le projet par défaut
$env:GCP_PROJECT_ID = "house-prices-mlops"  # Remplacer par votre ID
gcloud config set project $env:GCP_PROJECT_ID

# Lier la facturation (pour activer le crédit de $50)
gcloud beta billing accounts list
gcloud beta billing projects link $env:GCP_PROJECT_ID --billing-account=VOTRE_COMPTE_FACTURATION
```

## 🚢 Déploiement

### Option A: Script Automatisé (Recommandé)
```powershell
# Déployer en une commande
.\deploy-gcp.ps1
```

### Option B: Commandes Manuelles
```powershell
$PROJECT_ID = $env:GCP_PROJECT_ID
$REGION = "us-central1"
$SERVICE_NAME = "house-price-prediction"

# 1. Activer les APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# 2. Build et push de l'image
gcloud builds submit --tag "gcr.io/$PROJECT_ID/$SERVICE_NAME"

# 3. Déployer sur Cloud Run
gcloud run deploy $SERVICE_NAME `
  --image "gcr.io/$PROJECT_ID/$SERVICE_NAME" `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --min-instances 0 `
  --max-instances 10 `
  --port 8080
```

## 🧪 Tester le Déploiement

### 1. Récupérer l'URL du service
```powershell
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)"
Write-Host "Service URL: $SERVICE_URL"
```

### 2. Tester le health check
```powershell
Invoke-RestMethod "$SERVICE_URL/health"
```

### 3. Tester une prédiction
```powershell
$body = Get-Content test_prediction.json -Raw
Invoke-RestMethod -Uri "$SERVICE_URL/predict" -Method Post -Body $body -ContentType "application/json"
```

### 4. Accéder à la documentation
```
Ouvrir dans un navigateur: $SERVICE_URL/docs
```

## 💰 Estimation des Coûts

**Avec $50 de crédit Google Cloud:**

| Service | Coût/mois | Détails |
|---------|-----------|---------|
| Cloud Run | ~$5.00 | 2GB RAM, 2 vCPU, scaling 0-10 instances |
| Container Registry | $0.10 | Stockage de l'image Docker |
| Cloud Logging | $0.50 | Logs de l'application |
| Cloud Monitoring | $0.00 | Inclus dans la gratuité |
| **TOTAL** | **~$5.60/mois** | |

**Durée avec $50:** ~8-9 mois ✅

### Gratuit en dessous de:
- 2 millions de requêtes/mois
- 360,000 GB-secondes de mémoire
- 180,000 vCPU-secondes

## 📊 Monitoring Post-Déploiement

### Cloud Run Console
```powershell
# Ouvrir la console Cloud Run
gcloud run services describe $SERVICE_NAME --region $REGION
```

### Logs en temps réel
```powershell
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME"
```

### Métriques
```
Console: https://console.cloud.google.com/run?project=$PROJECT_ID
```

## 🔧 Commandes Utiles

### Voir les logs
```powershell
gcloud run services logs read $SERVICE_NAME --region $REGION --limit 50
```

### Mettre à jour le service
```powershell
# Après rebuild local
gcloud builds submit --tag "gcr.io/$PROJECT_ID/$SERVICE_NAME"
gcloud run deploy $SERVICE_NAME --image "gcr.io/$PROJECT_ID/$SERVICE_NAME" --region $REGION
```

### Arrêter le service (économiser le crédit)
```powershell
gcloud run services delete $SERVICE_NAME --region $REGION
```

### Voir l'utilisation du crédit
```
Console: https://console.cloud.google.com/billing
```

## 🎯 Prochaines Étapes

1. ✅ **Déployer** avec `.\deploy-gcp.ps1`
2. ✅ **Tester** l'API publique
3. ⚙️ **CI/CD** avec GitHub Actions (optionnel)
4. 📈 **Monitoring** avec Cloud Monitoring
5. 🔐 **Sécurité** avec IAM et API keys (production)

## 🆘 Troubleshooting

### Erreur: "billing not enabled"
```powershell
# Activer la facturation dans la console
gcloud beta billing projects link $env:GCP_PROJECT_ID --billing-account=ACCOUNT_ID
```

### Erreur: "permission denied"
```powershell
# Vérifier les permissions
gcloud projects get-iam-policy $env:GCP_PROJECT_ID
```

### Build trop lent
```powershell
# Utiliser l'image locale déjà buildée
docker tag house-prices-api:latest gcr.io/$PROJECT_ID/$SERVICE_NAME
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME
```

## 📞 Support

- Documentation Cloud Run: https://cloud.google.com/run/docs
- Pricing Calculator: https://cloud.google.com/products/calculator
- Support Community: https://stackoverflow.com/questions/tagged/google-cloud-run

---

**Projet:** MLOps House Prices Prediction  
**Image Docker:** house-prices-api:latest (2.94GB)  
**Modèle:** XGBoost Production v1.0  
**Performance:** 48ms/prediction ⚡
