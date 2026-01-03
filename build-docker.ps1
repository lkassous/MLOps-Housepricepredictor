# Script de build des images Docker
# Build l'image de production et teste le déploiement local

param(
    [switch]$NoBuild,
    [switch]$NoTest
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "BUILD DES IMAGES DOCKER - MLOPS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Verifier que Docker est disponible
Write-Host "`nVerification de Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "  Succes: Docker installe: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  Erreur: Docker n'est pas installe ou n'est pas demarre" -ForegroundColor Red
    Write-Host "  Veuillez installer Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Verifier que les fichiers necessaires existent
Write-Host "`nVerification des fichiers requis..." -ForegroundColor Yellow

$requiredFiles = @(
    "Dockerfile",
    "app.py",
    "requirements.txt",
    "mlflow.db"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  Succes: $file" -ForegroundColor Green
    } else {
        Write-Host "  Erreur: $file manquant" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "`nErreur: Fichiers manquants detectes. Impossible de continuer." -ForegroundColor Red
    exit 1
}

if (-not $NoBuild) {
    # Build de l'image Docker
    Write-Host "`n======================================================================" -ForegroundColor Cyan
    Write-Host "BUILD DE L'IMAGE DOCKER" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    
    $imageName = "house-prices-api"
    $imageTag = "latest"
    
    Write-Host "`nConstruction de l'image $imageName`:$imageTag..." -ForegroundColor Yellow
    Write-Host "(Cela peut prendre quelques minutes...)" -ForegroundColor Gray
    
    docker build -t "${imageName}:${imageTag}" . 2>&1 | ForEach-Object {
        if ($_ -match "Step \d+/\d+") {
            Write-Host $_ -ForegroundColor Cyan
        } elseif ($_ -match "Successfully") {
            Write-Host $_ -ForegroundColor Green
        } elseif ($_ -match "ERROR|error") {
            Write-Host $_ -ForegroundColor Red
        } else {
            Write-Host $_ -ForegroundColor Gray
        }
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSucces: Image construite avec succes: ${imageName}:${imageTag}" -ForegroundColor Green
    } else {
        Write-Host "`nErreur: Erreur lors du build de l'image" -ForegroundColor Red
        exit 1
    }
    
    # Afficher les images
    Write-Host "`nImages Docker disponibles:" -ForegroundColor Yellow
    docker images $imageName
}

if (-not $NoTest) {
    # Test de l'image
    Write-Host "`n======================================================================" -ForegroundColor Cyan
    Write-Host "TEST DE L'IMAGE DOCKER" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    
    $containerName = "house-prices-api-test"
    $port = 8080
    
    # Arrêter et supprimer le conteneur existant s'il existe
    Write-Host "`nNettoyage des conteneurs existants..." -ForegroundColor Yellow
    docker stop $containerName 2>$null | Out-Null
    docker rm $containerName 2>$null | Out-Null
    
    # Lancer le conteneur
    Write-Host "`nDémarrage du conteneur de test..." -ForegroundColor Yellow
    docker run -d --name $containerName -p "${port}:8080" house-prices-api:latest
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Succes: Conteneur demarre: $containerName" -ForegroundColor Green
        
        # Attendre que le service soit pret
        Write-Host "`nAttente du demarrage du service (30 secondes)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
        
        # Test du health endpoint
        Write-Host "`nTest de l'API..." -ForegroundColor Yellow
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$port/health" -Method Get -TimeoutSec 10
            $healthData = $response.Content | ConvertFrom-Json
            
            Write-Host "  Succes: Health check reussi" -ForegroundColor Green
            Write-Host "    Status: $($healthData.status)" -ForegroundColor Gray
            Write-Host "    Model loaded: $($healthData.model_loaded)" -ForegroundColor Gray
            Write-Host "    Uptime: $([math]::Round($healthData.uptime_seconds, 2))s" -ForegroundColor Gray
            
        } catch {
            Write-Host "  Erreur: Erreur lors du test de l'API: $($_.Exception.Message)" -ForegroundColor Red
        }
        
        # Afficher les logs
        Write-Host "`nDerniers logs du conteneur:" -ForegroundColor Yellow
        docker logs $containerName --tail 20
        
        # Options pour l'utilisateur
        Write-Host "`n======================================================================" -ForegroundColor Cyan
        Write-Host "CONTENEUR TEST EN COURS D'EXECUTION" -ForegroundColor Green
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "API accessible: http://localhost:$port" -ForegroundColor White
        Write-Host "Documentation Swagger: http://localhost:$port/docs" -ForegroundColor White
        Write-Host "Health check: http://localhost:$port/health" -ForegroundColor White
        Write-Host ""
        Write-Host "Commandes utiles:" -ForegroundColor Yellow
        Write-Host "  - Voir les logs: docker logs $containerName -f" -ForegroundColor Gray
        Write-Host "  - Arrêter: docker stop $containerName" -ForegroundColor Gray
        Write-Host "  - Supprimer: docker rm $containerName" -ForegroundColor Gray
        Write-Host ""
        
        # Demander si on veut arrêter le conteneur
        $response = Read-Host "Voulez-vous arrêter le conteneur de test? (O/N)"
        if ($response -eq "O" -or $response -eq "o" -or $response -eq "Y" -or $response -eq "y") {
            Write-Host "`nArret du conteneur..." -ForegroundColor Yellow
            docker stop $containerName | Out-Null
            docker rm $containerName | Out-Null
            Write-Host "  Succes: Conteneur arrete et supprime" -ForegroundColor Green
        } else {
            Write-Host "`nConteneur toujours actif. Arretez-le manuellement quand vous aurez termine." -ForegroundColor Yellow
        }
        
    } else {
        Write-Host "  Erreur: Erreur lors du demarrage du conteneur" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "BUILD DOCKER TERMINE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
