# Script pour déployer MLflow sur Google Cloud Run
# Exécuter depuis Cloud Shell ou localement avec gcloud configuré

param(
    [string]$ProjectId = "mlops-house-project",
    [string]$Region = "us-central1"
)

Write-Host "🚀 Déploiement du serveur MLflow sur Cloud Run" -ForegroundColor Cyan

# Variables
$ServiceName = "mlflow-server"
$ImageName = "us-central1-docker.pkg.dev/$ProjectId/docker-repo/mlflow-server:latest"

# Vérifier la configuration gcloud
Write-Host "📋 Vérification de la configuration GCP..." -ForegroundColor Yellow
gcloud config set project $ProjectId

# Construire l'image Docker
Write-Host "🐳 Construction de l'image Docker MLflow..." -ForegroundColor Yellow
docker build -f Dockerfile.mlflow -t $ImageName .

# Configurer Docker pour Artifact Registry
Write-Host "🔐 Configuration de Docker pour Artifact Registry..." -ForegroundColor Yellow
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# Pousser l'image
Write-Host "📤 Push de l'image vers Artifact Registry..." -ForegroundColor Yellow
docker push $ImageName

# Déployer sur Cloud Run
Write-Host "🚀 Déploiement sur Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --image=$ImageName `
    --region=$Region `
    --platform=managed `
    --allow-unauthenticated `
    --port=8080 `
    --memory=512Mi `
    --timeout=300 `
    --min-instances=0 `
    --max-instances=1

# Récupérer l'URL du service
$ServiceUrl = gcloud run services describe $ServiceName --region=$Region --format='value(status.url)'

Write-Host ""
Write-Host "✅ MLflow Server déployé avec succès!" -ForegroundColor Green
Write-Host "🌐 URL: $ServiceUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Prochaines étapes:" -ForegroundColor Yellow
Write-Host "1. Ajoutez ce secret dans GitHub:" -ForegroundColor White
Write-Host "   Nom: MLFLOW_TRACKING_URI" -ForegroundColor White
Write-Host "   Valeur: $ServiceUrl" -ForegroundColor White
Write-Host ""
Write-Host "2. Testez avec:" -ForegroundColor White
Write-Host "   `$env:MLFLOW_TRACKING_URI = '$ServiceUrl'" -ForegroundColor White
Write-Host "   python pipeline.py --data-path train.csv" -ForegroundColor White
