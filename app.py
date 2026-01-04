"""
FastAPI application pour servir le modèle MLflow en production
API RESTful avec endpoints pour prédictions, health check, et model info
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime
import uvicorn
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "HousePrices-TunedModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", None)  # Option pour charger directement par run_id

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialisation FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="API MLOps pour prédire les prix des maisons avec MLflow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_info = {}
label_encoders = {}
feature_names = []

# Pydantic models pour validation
class HouseFeatures(BaseModel):
    """Features d'une maison pour la prédiction"""
    MSSubClass: int = Field(..., description="Type de logement")
    MSZoning: str = Field(..., description="Zonage général")
    LotFrontage: float = Field(..., description="Pieds linéaires de rue")
    LotArea: int = Field(..., description="Superficie du terrain en pieds carrés")
    Street: str = Field(..., description="Type d'accès routier")
    Alley: Optional[str] = Field(None, description="Type d'accès à l'allée")
    LotShape: str = Field(..., description="Forme générale de la propriété")
    LandContour: str = Field(..., description="Planéité de la propriété")
    Utilities: str = Field(..., description="Type de services publics")
    LotConfig: str = Field(..., description="Configuration du lot")
    LandSlope: str = Field(..., description="Pente de la propriété")
    Neighborhood: str = Field(..., description="Quartier")
    Condition1: str = Field(..., description="Proximité routes/voies ferrées")
    Condition2: str = Field(..., description="Proximité routes/voies ferrées (si seconde)")
    BldgType: str = Field(..., description="Type de logement")
    HouseStyle: str = Field(..., description="Style de logement")
    OverallQual: int = Field(..., ge=1, le=10, description="Qualité générale")
    OverallCond: int = Field(..., ge=1, le=10, description="État général")
    YearBuilt: int = Field(..., description="Année de construction")
    YearRemodAdd: int = Field(..., description="Année de rénovation")
    RoofStyle: str = Field(..., description="Type de toit")
    RoofMatl: str = Field(..., description="Matériau du toit")
    Exterior1st: str = Field(..., description="Revêtement extérieur")
    Exterior2nd: str = Field(..., description="Revêtement extérieur (si plusieurs)")
    MasVnrType: Optional[str] = Field(None, description="Type de placage de maçonnerie")
    MasVnrArea: float = Field(..., description="Surface de placage de maçonnerie")
    ExterQual: str = Field(..., description="Qualité du matériau extérieur")
    ExterCond: str = Field(..., description="État du matériau extérieur")
    Foundation: str = Field(..., description="Type de fondation")
    BsmtQual: Optional[str] = Field(None, description="Qualité du sous-sol")
    BsmtCond: Optional[str] = Field(None, description="État du sous-sol")
    BsmtExposure: Optional[str] = Field(None, description="Exposition du sous-sol")
    BsmtFinType1: Optional[str] = Field(None, description="Finition du sous-sol zone 1")
    BsmtFinSF1: float = Field(..., description="Pieds carrés finis zone 1")
    BsmtFinType2: Optional[str] = Field(None, description="Finition du sous-sol zone 2")
    BsmtFinSF2: float = Field(..., description="Pieds carrés finis zone 2")
    BsmtUnfSF: float = Field(..., description="Pieds carrés non finis sous-sol")
    TotalBsmtSF: float = Field(..., description="Pieds carrés totaux sous-sol")
    Heating: str = Field(..., description="Type de chauffage")
    HeatingQC: str = Field(..., description="Qualité et état du chauffage")
    CentralAir: str = Field(..., description="Climatisation centrale")
    Electrical: Optional[str] = Field(None, description="Système électrique")
    FirstFlrSF: int = Field(..., alias="1stFlrSF", description="Pieds carrés 1er étage")
    SecondFlrSF: int = Field(..., alias="2ndFlrSF", description="Pieds carrés 2e étage")
    LowQualFinSF: int = Field(..., description="Pieds carrés finis basse qualité")
    GrLivArea: int = Field(..., description="Surface habitable hors sol")
    BsmtFullBath: int = Field(..., description="Salles de bain complètes sous-sol")
    BsmtHalfBath: int = Field(..., description="Demi-salles de bain sous-sol")
    FullBath: int = Field(..., description="Salles de bain complètes hors sol")
    HalfBath: int = Field(..., description="Demi-salles de bain hors sol")
    BedroomAbvGr: int = Field(..., description="Chambres hors sol")
    KitchenAbvGr: int = Field(..., description="Cuisines hors sol")
    KitchenQual: str = Field(..., description="Qualité de la cuisine")
    TotRmsAbvGrd: int = Field(..., description="Total pièces hors sol")
    Functional: str = Field(..., description="Fonctionnalité du logement")
    Fireplaces: int = Field(..., description="Nombre de foyers")
    FireplaceQu: Optional[str] = Field(None, description="Qualité du foyer")
    GarageType: Optional[str] = Field(None, description="Emplacement du garage")
    GarageYrBlt: Optional[float] = Field(None, description="Année construction garage")
    GarageFinish: Optional[str] = Field(None, description="Finition intérieure garage")
    GarageCars: int = Field(..., description="Capacité garage en voitures")
    GarageArea: int = Field(..., description="Surface garage en pieds carrés")
    GarageQual: Optional[str] = Field(None, description="Qualité du garage")
    GarageCond: Optional[str] = Field(None, description="État du garage")
    PavedDrive: str = Field(..., description="Allée pavée")
    WoodDeckSF: int = Field(..., description="Surface terrasse bois")
    OpenPorchSF: int = Field(..., description="Surface porche ouvert")
    EnclosedPorch: int = Field(..., description="Surface porche fermé")
    ThreeSsnPorch: int = Field(..., alias="3SsnPorch", description="Surface porche 3 saisons")
    ScreenPorch: int = Field(..., description="Surface porche grillagé")
    PoolArea: int = Field(..., description="Surface piscine")
    PoolQC: Optional[str] = Field(None, description="Qualité de la piscine")
    Fence: Optional[str] = Field(None, description="Qualité de la clôture")
    MiscFeature: Optional[str] = Field(None, description="Caractéristique diverse")
    MiscVal: int = Field(..., description="Valeur caractéristique diverse")
    MoSold: int = Field(..., ge=1, le=12, description="Mois de vente")
    YrSold: int = Field(..., description="Année de vente")
    SaleType: str = Field(..., description="Type de vente")
    SaleCondition: str = Field(..., description="Condition de vente")

    class Config:
        allow_population_by_field_name = True

class PredictionRequest(BaseModel):
    """Request pour prédiction unique ou batch"""
    houses: List[HouseFeatures] = Field(..., description="Liste de maisons à prédire")

class PredictionResponse(BaseModel):
    """Response avec prédictions"""
    predictions: List[float] = Field(..., description="Prix prédits")
    model_name: str = Field(..., description="Nom du modèle utilisé")
    model_version: str = Field(..., description="Version du modèle")
    prediction_time_ms: float = Field(..., description="Temps de prédiction en ms")

class HealthResponse(BaseModel):
    """Response du health check"""
    status: str = Field(..., description="État du service")
    model_loaded: bool = Field(..., description="Modèle chargé")
    model_name: str = Field(..., description="Nom du modèle")
    uptime_seconds: float = Field(..., description="Temps de fonctionnement")

class ModelInfoResponse(BaseModel):
    """Informations sur le modèle"""
    model_name: str
    model_version: str
    model_stage: str
    run_id: str
    metrics: Dict[str, float]
    loaded_at: str

# Variables globales pour monitoring
start_time = time.time()
prediction_count = 0
comparison_results = {}

def load_model():
    """Charge le modèle depuis model_artifacts (fallback sur pickle)"""
    global model, model_info, label_encoders, feature_names, comparison_results
    
    try:
        import pickle
        
        # Charger le modèle pickle
        with open('model_artifacts/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Charger les label encoders
        with open('model_artifacts/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Charger les feature names
        with open('model_artifacts/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Charger les infos du modèle
        model_info_path = 'model_artifacts/model_info.json'
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                saved_info = json.load(f)
            model_info = {
                "model_name": saved_info.get('model_name', 'Unknown'),
                "model_type": saved_info.get('model_type', 'Unknown'),
                "model_version": "1.0",
                "loaded_at": datetime.now().isoformat(),
                "trained_at": saved_info.get('timestamp', 'Unknown'),
                "load_method": "pickle_artifacts",
                "n_features": len(feature_names)
            }
        else:
            model_info = {
                "model_name": "Production-Model",
                "model_version": "1.0",
                "loaded_at": datetime.now().isoformat(),
                "load_method": "pickle_artifacts",
                "n_features": len(feature_names)
            }
        
        # Charger les résultats de comparaison
        comparison_path = 'model_artifacts/comparison_results.json'
        if os.path.exists(comparison_path):
            with open(comparison_path, 'r') as f:
                comparison_results = json.load(f)
            logger.info(f"Comparison results loaded: Best model = {comparison_results.get('best_model')}")
        
        logger.info("Modèle chargé depuis model_artifacts/")
        logger.info(f"Model name: {model_info.get('model_name')}")
        logger.info(f"Label encoders: {len(label_encoders)} colonnes")
        logger.info(f"Features: {len(feature_names)}")
        
        logger.info(f"Modèle chargé avec succès: {model_info}")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        logger.exception("Détails de l'erreur:")
        return False

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage - charge le modèle"""
    logger.info("Démarrage de l'API...")
    success = load_model()
    if not success:
        logger.warning("Le modèle n'a pas pu être chargé au démarrage")

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt"""
    logger.info("Arrêt de l'API...")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware pour logger toutes les requêtes"""
    start_time_req = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time_req) * 1000
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )
    
    return response

@app.get("/", tags=["General"], response_class=HTMLResponse)
async def root():
    """Page d'accueil avec interface web moderne"""
    static_index = os.path.join(os.path.dirname(__file__), "static", "index.html")
    
    # Si le fichier index.html existe, le servir
    if os.path.exists(static_index):
        with open(static_index, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    
    # Sinon, retourner le JSON de l'API
    return JSONResponse({
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "web_ui": "Install static files to see web interface"
    })

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check pour vérifier l'état du service"""
    uptime = time.time() - start_time
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_info.get('model_name', MODEL_NAME) if model_info else MODEL_NAME,
        "uptime_seconds": int(uptime),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/comparison", tags=["Model"])
async def get_comparison_results():
    """Récupère les résultats de comparaison des modèles du dernier pipeline run"""
    if not comparison_results:
        # Try to load from file if not in memory
        comparison_path = 'model_artifacts/comparison_results.json'
        if os.path.exists(comparison_path):
            with open(comparison_path, 'r') as f:
                return json.load(f)
        return {
            "error": "No comparison results available",
            "message": "Run the pipeline first to generate comparison data"
        }
    return comparison_results

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Récupère les informations sur le modèle chargé"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return ModelInfoResponse(**model_info)

@app.post("/reload-model", tags=["Model"])
async def reload_model():
    """Recharge le modèle depuis MLflow"""
    success = load_model()
    
    if success:
        return {"message": "Modèle rechargé avec succès", "model_info": model_info}
    else:
        raise HTTPException(status_code=500, detail="Échec du rechargement du modèle")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Prédit le prix de maisons
    
    Accepte une liste de maisons avec leurs caractéristiques
    et retourne les prix prédits
    """
    global prediction_count
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Veuillez réessayer plus tard."
        )
    
    try:
        start_pred = time.time()
        
        # Convertir les données en DataFrame
        houses_data = [house.dict(by_alias=True) for house in request.houses]
        df = pd.DataFrame(houses_data)
        
        # Preprocessing: Imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Remplir les valeurs manquantes
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Missing')
        
        # Label encoding avec les encoders sauvegardés
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                # Transformer les valeurs, gérer les nouvelles catégories
                df[col] = df[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        
        # S'assurer que toutes les colonnes nécessaires sont présentes
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Réordonner les colonnes selon l'ordre d'entraînement
        df = df[feature_names]
        
        # Faire les prédictions
        predictions = model.predict(df)
        predictions_list = predictions.tolist()
        
        # Calculer le temps de prédiction
        prediction_time = (time.time() - start_pred) * 1000
        
        # Incrémenter le compteur
        prediction_count += len(predictions_list)
        
        logger.info(
            f"Prédiction réussie pour {len(predictions_list)} maisons - "
            f"Temps: {prediction_time:.2f}ms"
        )
        
        return PredictionResponse(
            predictions=predictions_list,
            model_name=model_info["model_name"],
            model_version=str(model_info.get("model_version", "1.0")),
            prediction_time_ms=prediction_time
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Métriques de monitoring de l'API"""
    uptime = time.time() - start_time
    
    return {
        "uptime_seconds": uptime,
        "total_predictions": prediction_count,
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_version": model_info.get("model_version", "unknown") if model_info else "unknown"
    }

if __name__ == "__main__":
    # Configuration pour développement local
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=True,
        log_level="info"
    )
