#!/bin/bash
# Script de déploiement sur Google Cloud Run

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="house-price-prediction"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "======================================================================"
echo "DÉPLOIEMENT MLOPS SUR GOOGLE CLOUD RUN"
echo "======================================================================"

# 1. Vérifier que gcloud est configuré
echo "1. Vérification de la configuration gcloud..."
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI n'est pas installé"
    echo "Installation: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

gcloud config set project ${PROJECT_ID}
echo "✅ Projet configuré: ${PROJECT_ID}"

# 2. Activer les APIs nécessaires
echo ""
echo "2. Activation des APIs Google Cloud..."
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com

echo "✅ APIs activées"

# 3. Build de l'image Docker avec Cloud Build
echo ""
echo "3. Build de l'image Docker sur Cloud Build..."
gcloud builds submit --tag ${IMAGE_NAME}:latest .

echo "✅ Image Docker construite: ${IMAGE_NAME}:latest"

# 4. Déployer sur Cloud Run
echo ""
echo "4. Déploiement sur Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 \
    --port 8080 \
    --set-env-vars "MODEL_NAME=HousePrices-TunedModel,MODEL_STAGE=Production"

echo "✅ Service déployé sur Cloud Run"

# 5. Récupérer l'URL du service
echo ""
echo "5. Récupération de l'URL du service..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo "======================================================================"
echo "✅ DÉPLOIEMENT TERMINÉ AVEC SUCCÈS!"
echo "======================================================================"
echo ""
echo "🌐 URL de l'API: ${SERVICE_URL}"
echo "📚 Documentation: ${SERVICE_URL}/docs"
echo "💚 Health check: ${SERVICE_URL}/health"
echo ""
echo "📊 Test de l'API:"
echo "curl ${SERVICE_URL}/health"
echo ""
echo "======================================================================"
