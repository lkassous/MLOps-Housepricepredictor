# Script de démarrage de MLflow UI
# Lance l'interface MLflow sur http://localhost:5000

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Démarrage de MLflow UI" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier si MLflow est installé
try {
    $mlflowVersion = python -c "import mlflow; print(mlflow.__version__)" 2>&1
    Write-Host "✓ MLflow version: $mlflowVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ MLflow n'est pas installé!" -ForegroundColor Red
    Write-Host "  Exécutez d'abord: .\install.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Démarrage du serveur MLflow..." -ForegroundColor Yellow
Write-Host ""
Write-Host "📊 Interface MLflow sera accessible sur:" -ForegroundColor Green
Write-Host "   http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚡ Pour arrêter le serveur: Ctrl+C" -ForegroundColor Yellow
Write-Host ""
Write-Host "Chargement en cours..." -ForegroundColor White
Write-Host ""

# Démarrer MLflow UI
mlflow ui

Write-Host ""
Write-Host "MLflow UI arrêté." -ForegroundColor Yellow
