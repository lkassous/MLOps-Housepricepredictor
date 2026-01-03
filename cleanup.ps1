# Script de nettoyage du projet
# Supprime les fichiers temporaires, caches, et fichiers générés

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "NETTOYAGE DU PROJET MLOPS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$itemsToClean = @(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache",
    ".coverage",
    "htmlcov",
    "*.egg-info",
    ".ipynb_checkpoints",
    "*_results.csv",
    "submission_*.csv",
    "feature_importance.csv",
    "hyperparameter_tuning_results.csv",
    "mlflow_analysis_results.png",
    "mlflow_correlation_heatmap.png",
    ".env"
)

Write-Host "`nSuppression des fichiers temporaires et caches..." -ForegroundColor Yellow

foreach ($pattern in $itemsToClean) {
    $items = Get-ChildItem -Path . -Include $pattern -Recurse -Force -ErrorAction SilentlyContinue
    
    if ($items) {
        foreach ($item in $items) {
            try {
                Remove-Item $item.FullName -Recurse -Force -ErrorAction Stop
                Write-Host "  Succes: Supprime: $($item.Name)" -ForegroundColor Green
            } catch {
                Write-Host "  Erreur: $($item.Name) - $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}

Write-Host "  Succes: Nettoyage des fichiers termine" -ForegroundColor Green

# Nettoyer les images Docker orphelines (optionnel)
Write-Host "`nNettoyage des images Docker orphelines..." -ForegroundColor Yellow
try {
    docker system prune -f 2>$null | Out-Null
    Write-Host "  Succes: Images Docker orphelines supprimees" -ForegroundColor Green
} catch {
    Write-Host "  Info: Docker non demarre ou non installe" -ForegroundColor Gray
}

# Créer les répertoires nécessaires
Write-Host "`nCréation des répertoires nécessaires..." -ForegroundColor Yellow

$directories = @(
    "tests",
    ".github/workflows",
    "logs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Succes: Cree: $dir" -ForegroundColor Green
    } else {
        Write-Host "  Info: Existe deja: $dir" -ForegroundColor Gray
    }
}

# Creer __init__.py pour tests
if (-not (Test-Path "tests/__init__.py")) {
    New-Item -ItemType File -Path "tests/__init__.py" -Force | Out-Null
    Write-Host "  Succes: Cree: tests/__init__.py" -ForegroundColor Green
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "NETTOYAGE TERMINE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "`nProjet pret pour le build Docker!" -ForegroundColor White
