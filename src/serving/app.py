"""
FastAPI Application for Phishing Detection
Serves trained models via REST API
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, load_config
from serving.schema import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, ModelsListResponse, ModelInfo
)

# Initialize logger
logger = setup_logger(__name__, "logs/api.log")

# Load configuration
config = load_config("src/config/config.yaml")
api_config = config['api']

# Initialize FastAPI app
app = FastAPI(
    title=api_config['title'],
    description=api_config['description'],
    version=api_config['version']
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
MODELS = {}
DEFAULT_MODEL = "random_forest"
FEATURE_NAMES = None

def load_models():
    """Load all trained models"""
    global MODELS, FEATURE_NAMES
    
    models_dir = Path(config['training']['save_model_path'])
    logger.info(f"Loading models from: {models_dir}")
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return False
    
    # Load all *_latest.pkl models
    for model_file in models_dir.glob("*_latest.pkl"):
        model_name = model_file.stem.replace("_latest", "")
        try:
            model = joblib.load(model_file)
            MODELS[model_name] = model
            logger.info(f"✓ Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"✗ Error loading {model_name}: {str(e)}")
    
    if not MODELS:
        logger.error("No models loaded!")
        return False
    
    # Load feature names from a sample data file
    try:
        feature_data = pd.read_csv(config['data']['feature_data_path'])
        FEATURE_NAMES = [col for col in feature_data.columns if col != config['data']['target_column']]
        logger.info(f"✓ Loaded {len(FEATURE_NAMES)} feature names")
    except Exception as e:
        logger.warning(f"Could not load feature names: {str(e)}")
    
    logger.info(f"✓ Successfully loaded {len(MODELS)} models: {list(MODELS.keys())}")
    return True

def engineer_features(df):
    """Apply feature engineering to input data"""
    # Drop features that were removed during training
    drop_features = config['features']['drop_features']
    for feat in drop_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])
    
    # Create engineered features (same as in build_features.py)
    df['SSLDomainTrust'] = df['SSLfinal_State'] * df['Domain_registeration_length']
    df['URLSuspicionScore'] = (df['having_IP_Address'] + df['URL_Length'] + 
                               df['having_At_Symbol'] + df['Prefix_Suffix']) / 4.0
    df['ContentCredibility'] = (df['URL_of_Anchor'] + df['Links_in_tags'] + df['SFH']) / 3.0
    df['DomainReputation'] = (df['age_of_domain'] + df['DNSRecord'] + df['web_traffic']) / 3.0
    
    security_features = ['SSLfinal_State', 'Domain_registeration_length', 
                        'HTTPS_token', 'age_of_domain', 'DNSRecord']
    df['SecurityFeaturesCount'] = df[security_features].apply(lambda x: (x == 1).sum(), axis=1)
    
    suspicious_features = ['having_IP_Address', 'having_At_Symbol', 'Prefix_Suffix',
                          'having_Sub_Domain', 'Request_URL', 'Abnormal_URL', 'Redirect']
    df['SuspiciousFeaturesCount'] = df[suspicious_features].apply(lambda x: (x == -1).sum(), axis=1)
    
    df['SSLAnchorInteraction'] = df['SSLfinal_State'] * df['URL_of_Anchor']
    
    return df

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("="*60)
    logger.info("STARTING FASTAPI APPLICATION")
    logger.info("="*60)
    
    success = load_models()
    if not success:
        logger.error("Failed to load models. API will not work properly.")
    else:
        logger.info("✓ API ready to serve predictions")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Network Security - Phishing Detection API",
        "version": api_config['version'],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODELS else "unhealthy",
        version=api_config['version'],
        models_loaded=list(MODELS.keys()),
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """List all available models"""
    models_info = []
    
    for model_name, model in MODELS.items():
        model_info = ModelInfo(
            model_name=model_name,
            model_type=type(model).__name__
        )
        models_info.append(model_info)
    
    return ModelsListResponse(
        models=models_info,
        total_models=len(MODELS),
        default_model=DEFAULT_MODEL
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    try:
        # Get model
        model_name = request.model_name
        if model_name not in MODELS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found. Available models: {list(MODELS.keys())}"
            )
        
        model = MODELS[model_name]
        
        # Convert input to DataFrame
        features_dict = request.features.dict()
        df = pd.DataFrame([features_dict])
        
        # Apply feature engineering
        df = engineer_features(df)
        
        # Ensure correct column order
        if FEATURE_NAMES:
            df = df[FEATURE_NAMES]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get probability if available
        probability = None
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            probability = float(proba[1])  # Probability of legitimate (class 1)
            confidence = float(max(proba))
        
        # Map prediction to label
        prediction_label = "Legitimate" if prediction == 1 else "Phishing"
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            probability=probability,
            confidence=confidence,
            model_used=model_name,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    try:
        # Get model
        model_name = request.model_name
        if model_name not in MODELS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        model = MODELS[model_name]
        
        # Convert input to DataFrame
        features_list = [f.dict() for f in request.features]
        df = pd.DataFrame(features_list)
        
        # Apply feature engineering
        df = engineer_features(df)
        
        # Ensure correct column order
        if FEATURE_NAMES:
            df = df[FEATURE_NAMES]
        
        # Make predictions
        predictions = model.predict(df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)
        
        # Create response
        prediction_responses = []
        timestamp = datetime.now().isoformat()
        
        for i, pred in enumerate(predictions):
            prob = float(probabilities[i][1]) if probabilities is not None else None
            conf = float(max(probabilities[i])) if probabilities is not None else None
            
            prediction_responses.append(
                PredictionResponse(
                    prediction=int(pred),
                    prediction_label="Legitimate" if pred == 1 else "Phishing",
                    probability=prob,
                    confidence=conf,
                    model_used=model_name,
                    timestamp=timestamp
                )
            )
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_predictions=len(predictions),
            model_used=model_name,
            timestamp=timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/{model_name}/info", tags=["Models"])
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    if model_name not in MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    model = MODELS[model_name]
    
    info = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "parameters": model.get_params() if hasattr(model, 'get_params') else {},
        "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else None
    }
    
    return info

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=api_config['host'],
        port=api_config['port'],
        reload=api_config['reload']
    )
