# Script d'Installation MLOps - House Prices Project
# Ce script installe Python et toutes les dépendances nécessaires

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Installation MLOps - House Prices Project" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Fonction pour vérifier si une commande existe
function Test-Command {
    param($cmdname)
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Étape 1: Vérifier si Python est installé
Write-Host "Étape 1: Vérification de Python..." -ForegroundColor Yellow
if (Test-Command python) {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python est installé: $pythonVersion" -ForegroundColor Green
    
    # Vérifier la version
    $versionNumber = [regex]::Match($pythonVersion, '\d+\.\d+').Value
    if ([version]$versionNumber -ge [version]"3.8") {
        Write-Host "✓ Version Python compatible (>= 3.8)" -ForegroundColor Green
    } else {
        Write-Host "✗ Version Python trop ancienne. Veuillez installer Python 3.10+" -ForegroundColor Red
        Write-Host "  Téléchargez depuis: https://www.python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✗ Python n'est pas installé!" -ForegroundColor Red
    Write-Host ""
    Write-Host "INSTRUCTIONS D'INSTALLATION:" -ForegroundColor Yellow
    Write-Host "1. Allez sur https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. Téléchargez Python 3.10 ou plus récent" -ForegroundColor White
    Write-Host "3. Pendant l'installation, COCHEZ 'Add Python to PATH'" -ForegroundColor Red
    Write-Host "4. Redémarrez PowerShell après l'installation" -ForegroundColor White
    Write-Host "5. Relancez ce script" -ForegroundColor White
    Write-Host ""
    
    # Proposer d'ouvrir le navigateur
    $response = Read-Host "Voulez-vous ouvrir le site de téléchargement maintenant? (O/N)"
    if ($response -eq "O" -or $response -eq "o") {
        Start-Process "https://www.python.org/downloads/"
    }
    exit 1
}

Write-Host ""

# Étape 2: Vérifier pip
Write-Host "Étape 2: Vérification de pip..." -ForegroundColor Yellow
try {
    $pipVersion = python -m pip --version 2>&1
    Write-Host "✓ pip est installé: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ pip n'est pas disponible" -ForegroundColor Red
    Write-Host "  Installation de pip..." -ForegroundColor Yellow
    python -m ensurepip --upgrade
}

Write-Host ""

# Étape 3: Mise à jour de pip
Write-Host "Étape 3: Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "✓ pip mis à jour" -ForegroundColor Green

Write-Host ""

# Étape 4: Installation des dépendances
Write-Host "Étape 4: Installation des dépendances MLOps..." -ForegroundColor Yellow
Write-Host "  Cela peut prendre plusieurs minutes..." -ForegroundColor White
Write-Host ""

if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Toutes les dépendances ont été installées avec succès!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "✗ Erreur lors de l'installation des dépendances" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✗ Le fichier requirements.txt n'a pas été trouvé" -ForegroundColor Red
    Write-Host "  Assurez-vous d'être dans le bon dossier" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Étape 5: Vérification de l'installation de MLflow
Write-Host "Étape 5: Vérification de MLflow..." -ForegroundColor Yellow
try {
    $mlflowVersion = python -c "import mlflow; print(mlflow.__version__)" 2>&1
    Write-Host "✓ MLflow version: $mlflowVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ MLflow n'est pas correctement installé" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Étape 6: Vérification des autres bibliothèques essentielles
Write-Host "Étape 6: Vérification des bibliothèques essentielles..." -ForegroundColor Yellow

$libraries = @(
    @{Name="pandas"; Import="pandas"},
    @{Name="numpy"; Import="numpy"},
    @{Name="scikit-learn"; Import="sklearn"},
    @{Name="xgboost"; Import="xgboost"},
    @{Name="lightgbm"; Import="lightgbm"}
)

foreach ($lib in $libraries) {
    try {
        $version = python -c "import $($lib.Import); print($($lib.Import).__version__)" 2>&1
        Write-Host "  ✓ $($lib.Name): $version" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ $($lib.Name) non trouvé" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Installation Terminée!" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "PROCHAINES ÉTAPES:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Lire le guide complet:" -ForegroundColor White
Write-Host "   code GUIDE_MLOPS_FR.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Démarrage rapide du pipeline:" -ForegroundColor White
Write-Host "   python src/quick_start.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Ou exécuter étape par étape:" -ForegroundColor White
Write-Host "   python src/data_preparation.py" -ForegroundColor Cyan
Write-Host "   python src/train_models.py" -ForegroundColor Cyan
Write-Host "   python src/register_model.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Visualiser les résultats dans MLflow UI:" -ForegroundColor White
Write-Host "   mlflow ui" -ForegroundColor Cyan
Write-Host "   Puis ouvrir: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Bon apprentissage automatique! 🚀" -ForegroundColor Green
Write-Host ""
