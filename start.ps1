# Startup script for Windows (PowerShell)

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  HOUSE PRICES MLOPS PROJECT - DOCKER STARTUP" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "`n✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "`n❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Build Docker images
Write-Host "`n📦 Building Docker images..." -ForegroundColor Yellow
docker-compose build

# Start services
Write-Host "`n🚀 Starting MLflow services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to be ready
Write-Host "`n⏳ Waiting for services to start (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if services are running
Write-Host "`n🔍 Checking service status..." -ForegroundColor Yellow
docker-compose ps

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "  ✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan

Write-Host "`n📊 MLflow UI is available at: http://localhost:5000" -ForegroundColor Green
Write-Host "`nTo run the training pipeline, execute:" -ForegroundColor Yellow
Write-Host "  docker-compose exec training python src/train_models.py" -ForegroundColor White
Write-Host "`nOr use the quick start script:" -ForegroundColor Yellow
Write-Host "  docker-compose exec training python src/quick_start.py" -ForegroundColor White
Write-Host "`nTo view logs:" -ForegroundColor Yellow
Write-Host "  docker-compose logs -f" -ForegroundColor White
Write-Host "`nTo stop services:" -ForegroundColor Yellow
Write-Host "  docker-compose down" -ForegroundColor White
Write-Host ""
