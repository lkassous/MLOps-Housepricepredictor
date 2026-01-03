#!/bin/bash

# Startup script for the MLOps project

echo "========================================================================"
echo "  HOUSE PRICES MLOPS PROJECT - DOCKER STARTUP"
echo "========================================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Build Docker images
echo ""
echo "📦 Building Docker images..."
docker-compose build

# Start services
echo ""
echo "🚀 Starting MLflow services..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start (10 seconds)..."
sleep 10

# Check if services are running
echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "========================================================================"
echo "  ✅ SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 MLflow UI is available at: http://localhost:5000"
echo ""
echo "To run the training pipeline, execute:"
echo "  docker-compose exec training python src/train_models.py"
echo ""
echo "Or use the quick start script:"
echo "  docker-compose exec training python src/quick_start.py"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
