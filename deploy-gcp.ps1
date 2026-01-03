# Script PowerShell de déploiement sur Google Cloud Run

param(
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$Region = "us-central1",
    [string]$ServiceName = "house-price-prediction"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "DÉPLOIEMENT MLOPS SUR GOOGLE CLOUD RUN" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$ImageName = "gcr.io/$ProjectId/$ServiceName"

# 1. Vérifier que gcloud est configuré
Write-Host "`n1. Vérification de la configuration gcloud..." -ForegroundColor Yellow

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Host "❌ gcloud CLI n'est pas installé" -ForegroundColor Red
    Write-Host "Installation: https://cloud.google.com/sdk/docs/install" -ForegroundColor Red
    exit 1
}

gcloud config set project $ProjectId
Write-Host "✅ Projet configuré: $ProjectId" -ForegroundColor Green

# 2. Activer les APIs nécessaires
Write-Host "`n2. Activation des APIs Google Cloud..." -ForegroundColor Yellow

$apis = @(
    "run.googleapis.com",
    "containerregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com"
)

foreach ($api in $apis) {
    gcloud services enable $api
}

Write-Host "✅ APIs activées" -ForegroundColor Green

# 3. Build de l'image Docker avec Cloud Build
Write-Host "`n3. Build de l'image Docker sur Cloud Build..." -ForegroundColor Yellow
gcloud builds submit --tag "${ImageName}:latest" .

Write-Host "✅ Image Docker construite: ${ImageName}:latest" -ForegroundColor Green

# 4. Déployer sur Cloud Run
Write-Host "`n4. Déploiement sur Cloud Run..." -ForegroundColor Yellow

gcloud run deploy $ServiceName `
    --image "${ImageName}:latest" `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --min-instances 0 `
    --max-instances 10 `
    --port 8080 `
    --set-env-vars "MODEL_NAME=HousePrices-TunedModel,MODEL_STAGE=Production"

Write-Host "✅ Service déployé sur Cloud Run" -ForegroundColor Green

# 5. Récupérer l'URL du service
Write-Host "`n5. Récupération de l'URL du service..." -ForegroundColor Yellow

$ServiceUrl = gcloud run services describe $ServiceName `
    --platform managed `
    --region $Region `
    --format 'value(status.url)'

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "✅ DÉPLOIEMENT TERMINÉ AVEC SUCCÈS!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 URL de l'API: $ServiceUrl" -ForegroundColor White
Write-Host "📚 Documentation: $ServiceUrl/docs" -ForegroundColor White
Write-Host "💚 Health check: $ServiceUrl/health" -ForegroundColor White
Write-Host ""
Write-Host "📊 Test de l'API:" -ForegroundColor Yellow
Write-Host "Invoke-WebRequest -Uri '$ServiceUrl/health' | ConvertFrom-Json" -ForegroundColor Gray
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
