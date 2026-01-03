#!/usr/bin/env pwsh
# Script de vérification des prérequis Google Cloud

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "VÉRIFICATION DES PRÉREQUIS GOOGLE CLOUD" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$allGood = $true

# 1. Vérifier Google Cloud SDK
Write-Host "`n1. Google Cloud SDK..." -NoNewline
if (Get-Command gcloud -ErrorAction SilentlyContinue) {
    $version = (gcloud version --format="value(core)").Trim()
    Write-Host " ✅ Installé (v$version)" -ForegroundColor Green
} else {
    Write-Host " ❌ Non installé" -ForegroundColor Red
    Write-Host "   Installation: https://cloud.google.com/sdk/docs/install-sdk#windows" -ForegroundColor Yellow
    Write-Host "   Ou avec Chocolatey: choco install gcloudsdk" -ForegroundColor Yellow
    $allGood = $false
}

# 2. Vérifier authentification
Write-Host "`n2. Authentification Google Cloud..." -NoNewline
try {
    $account = (gcloud config get-value account 2>$null).Trim()
    if ($account) {
        Write-Host " ✅ Connecté ($account)" -ForegroundColor Green
    } else {
        Write-Host " ⚠️ Non authentifié" -ForegroundColor Yellow
        Write-Host "   Commande: gcloud auth login" -ForegroundColor Yellow
        $allGood = $false
    }
} catch {
    Write-Host " ❌ Erreur" -ForegroundColor Red
    $allGood = $false
}

# 3. Vérifier projet
Write-Host "`n3. Projet Google Cloud..." -NoNewline
try {
    $project = (gcloud config get-value project 2>$null).Trim()
    if ($project) {
        Write-Host " ✅ Configuré ($project)" -ForegroundColor Green
        $env:GCP_PROJECT_ID = $project
    } else {
        Write-Host " ⚠️ Aucun projet configuré" -ForegroundColor Yellow
        Write-Host "   Option 1: gcloud config set project VOTRE_PROJECT_ID" -ForegroundColor Yellow
        Write-Host "   Option 2: gcloud projects create house-prices-mlops" -ForegroundColor Yellow
        $allGood = $false
    }
} catch {
    Write-Host " ❌ Erreur" -ForegroundColor Red
    $allGood = $false
}

# 4. Vérifier facturation
Write-Host "`n4. Facturation activée..." -NoNewline
if ($project) {
    try {
        $billing = gcloud beta billing projects describe $project --format="value(billingAccountName)" 2>$null
        if ($billing) {
            Write-Host " ✅ Activée" -ForegroundColor Green
        } else {
            Write-Host " ⚠️ Non activée" -ForegroundColor Yellow
            Write-Host "   1. Aller sur https://console.cloud.google.com/billing" -ForegroundColor Yellow
            Write-Host "   2. Lier votre compte avec crédit $50" -ForegroundColor Yellow
            Write-Host "   3. gcloud beta billing projects link $project --billing-account=ACCOUNT_ID" -ForegroundColor Yellow
            $allGood = $false
        }
    } catch {
        Write-Host " ⚠️ Impossible de vérifier" -ForegroundColor Yellow
    }
} else {
    Write-Host " ⏭️ Projet non configuré" -ForegroundColor Gray
}

# 5. Vérifier Docker
Write-Host "`n5. Docker..." -NoNewline
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $dockerVersion = (docker --version).Split()[2].TrimEnd(',')
    Write-Host " ✅ Installé (v$dockerVersion)" -ForegroundColor Green
} else {
    Write-Host " ❌ Non installé" -ForegroundColor Red
    $allGood = $false
}

# 6. Vérifier l'image Docker locale
Write-Host "`n6. Image Docker house-prices-api..." -NoNewline
$image = docker images house-prices-api:latest --format "{{.Repository}}:{{.Tag}}" 2>$null
if ($image) {
    $size = docker images house-prices-api:latest --format "{{.Size}}" 2>$null
    Write-Host " ✅ Prête ($size)" -ForegroundColor Green
} else {
    Write-Host " ❌ Non trouvée" -ForegroundColor Red
    Write-Host "   Commande: docker build -t house-prices-api:latest ." -ForegroundColor Yellow
    $allGood = $false
}

# 7. Vérifier APIs nécessaires
if ($project) {
    Write-Host "`n7. APIs Google Cloud nécessaires..." -NoNewline
    $requiredApis = @("run.googleapis.com", "cloudbuild.googleapis.com", "containerregistry.googleapis.com")
    $enabledApis = gcloud services list --enabled --format="value(config.name)" 2>$null
    
    $missingApis = @()
    foreach ($api in $requiredApis) {
        if ($enabledApis -notcontains $api) {
            $missingApis += $api
        }
    }
    
    if ($missingApis.Count -eq 0) {
        Write-Host " ✅ Toutes activées" -ForegroundColor Green
    } else {
        Write-Host " ⚠️ ${missingApis.Count} à activer" -ForegroundColor Yellow
        foreach ($api in $missingApis) {
            Write-Host "   - $api" -ForegroundColor Yellow
        }
        Write-Host "   Auto-activation lors du déploiement" -ForegroundColor Cyan
    }
}

# Résumé
Write-Host "`n======================================================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "✅ TOUS LES PRÉREQUIS SONT SATISFAITS" -ForegroundColor Green
    Write-Host "`nVous pouvez déployer avec: .\deploy-gcp.ps1" -ForegroundColor Green
    Write-Host "Ou suivre le guide: DEPLOYMENT_GUIDE.md" -ForegroundColor Cyan
} else {
    Write-Host "⚠️ CERTAINS PRÉREQUIS MANQUENT" -ForegroundColor Yellow
    Write-Host "`nConsultez le guide: DEPLOYMENT_GUIDE.md" -ForegroundColor Cyan
    Write-Host "Ou installez les prérequis manquants ci-dessus" -ForegroundColor Yellow
}
Write-Host "======================================================================" -ForegroundColor Cyan

# Afficher le crédit disponible si possible
if ($project -and $allGood) {
    Write-Host "`n💰 Estimation des coûts:" -ForegroundColor Cyan
    Write-Host "   - Cloud Run: environ 5 dollars/mois" -ForegroundColor White
    Write-Host "   - Storage & Logs: environ 0.60 dollars/mois" -ForegroundColor White
    Write-Host "   - TOTAL: environ 5.60 dollars/mois" -ForegroundColor White
    Write-Host "   - Duree avec 50 dollars de credit: environ 8-9 mois" -ForegroundColor Green
}
