"""
Tests pour l'API FastAPI
"""
import pytest
from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)

def test_root():
    """Test de la route racine"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "House Price Prediction API"

def test_health_check():
    """Test du health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "uptime_seconds" in data

def test_model_info():
    """Test de récupération des infos du modèle"""
    response = client.get("/model-info")
    # Peut être 200 ou 503 selon si le modèle est chargé
    assert response.status_code in [200, 503]

def test_metrics():
    """Test des métriques"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "total_predictions" in data
    assert "model_loaded" in data

def test_predict_endpoint_structure():
    """Test de la structure de l'endpoint de prédiction"""
    # Données de test minimales
    test_data = {
        "houses": [{
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotFrontage": 65.0,
            "LotArea": 8450,
            "Street": "Pave",
            "LotShape": "Reg",
            "LandContour": "Lvl",
            "Utilities": "AllPub",
            "LotConfig": "Inside",
            "LandSlope": "Gtl",
            "Neighborhood": "CollgCr",
            "Condition1": "Norm",
            "Condition2": "Norm",
            "BldgType": "1Fam",
            "HouseStyle": "2Story",
            "OverallQual": 7,
            "OverallCond": 5,
            "YearBuilt": 2003,
            "YearRemodAdd": 2003,
            "RoofStyle": "Gable",
            "RoofMatl": "CompShg",
            "Exterior1st": "VinylSd",
            "Exterior2nd": "VinylSd",
            "MasVnrArea": 196.0,
            "ExterQual": "Gd",
            "ExterCond": "TA",
            "Foundation": "PConc",
            "BsmtFinSF1": 706.0,
            "BsmtFinSF2": 0.0,
            "BsmtUnfSF": 150.0,
            "TotalBsmtSF": 856.0,
            "Heating": "GasA",
            "HeatingQC": "Ex",
            "CentralAir": "Y",
            "1stFlrSF": 856,
            "2ndFlrSF": 854,
            "LowQualFinSF": 0,
            "GrLivArea": 1710,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "FullBath": 2,
            "HalfBath": 1,
            "BedroomAbvGr": 3,
            "KitchenAbvGr": 1,
            "KitchenQual": "Gd",
            "TotRmsAbvGrd": 8,
            "Functional": "Typ",
            "Fireplaces": 0,
            "GarageCars": 2,
            "GarageArea": 548,
            "PavedDrive": "Y",
            "WoodDeckSF": 0,
            "OpenPorchSF": 61,
            "EnclosedPorch": 0,
            "3SsnPorch": 0,
            "ScreenPorch": 0,
            "PoolArea": 0,
            "MiscVal": 0,
            "MoSold": 2,
            "YrSold": 2008,
            "SaleType": "WD",
            "SaleCondition": "Normal"
        }]
    }
    
    response = client.post("/predict", json=test_data)
    
    # Le modèle peut ne pas être chargé dans les tests
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "model_name" in data
        assert "prediction_time_ms" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1
    elif response.status_code == 503:
        # Acceptable si le modèle n'est pas disponible dans l'environnement de test
        assert "Modèle non disponible" in response.json()["detail"]

def test_invalid_prediction_request():
    """Test avec des données invalides"""
    invalid_data = {
        "houses": [{
            "invalid_field": "value"
        }]
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_docs_available():
    """Test que la documentation Swagger est accessible"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_redoc_available():
    """Test que ReDoc est accessible"""
    response = client.get("/redoc")
    assert response.status_code == 200
