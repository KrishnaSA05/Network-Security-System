"""
FastAPI Application for Phishing Detection
Serves the best trained model via REST API
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger, load_config

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

# Global variables
BEST_MODEL = None
MODEL_METADATA = None
FEATURE_NAMES = None


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class FeaturesInput(BaseModel):
    """Input features for prediction"""
    having_IP_Address: int = Field(..., ge=-1, le=1)
    URL_Length: int = Field(..., ge=-1, le=1)
    Shortining_Service: int = Field(..., ge=-1, le=1)
    having_At_Symbol: int = Field(..., ge=-1, le=1)
    double_slash_redirecting: int = Field(..., ge=-1, le=1)
    Prefix_Suffix: int = Field(..., ge=-1, le=1)
    having_Sub_Domain: int = Field(..., ge=-1, le=1)
    SSLfinal_State: int = Field(..., ge=-1, le=1)
    Domain_registeration_length: int = Field(..., ge=-1, le=1)
    Favicon: int = Field(..., ge=-1, le=1)
    port: int = Field(..., ge=-1, le=1)
    HTTPS_token: int = Field(..., ge=-1, le=1)
    Request_URL: int = Field(..., ge=-1, le=1)
    URL_of_Anchor: int = Field(..., ge=-1, le=1)
    Links_in_tags: int = Field(..., ge=-1, le=1)
    SFH: int = Field(..., ge=-1, le=1)
    Submitting_to_email: int = Field(..., ge=-1, le=1)
    Abnormal_URL: int = Field(..., ge=-1, le=1)
    Redirect: int = Field(..., ge=0, le=1)
    on_mouseover: int = Field(..., ge=-1, le=1)
    RightClick: int = Field(..., ge=-1, le=1)
    popUpWidnow: int = Field(..., ge=-1, le=1)
    Iframe: int = Field(..., ge=-1, le=1)
    age_of_domain: int = Field(..., ge=-1, le=1)
    DNSRecord: int = Field(..., ge=-1, le=1)
    web_traffic: int = Field(..., ge=-1, le=1)
    Page_Rank: int = Field(..., ge=-1, le=1)
    Google_Index: int = Field(..., ge=-1, le=1)
    Links_pointing_to_page: int = Field(..., ge=-1, le=1)
    Statistical_report: int = Field(..., ge=-1, le=1)


class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    features: FeaturesInput


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction"""
    features: List[FeaturesInput]


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    prediction: int
    prediction_label: str
    probability: Optional[float] = None
    confidence: Optional[float] = None
    model_used: str
    model_accuracy: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction"""
    predictions: List[PredictionResponse]
    total_predictions: int
    model_used: str
    timestamp: str


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_accuracy: Optional[float] = None
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    trained_at: str


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_best_model():
    """Load the best trained model and its metadata"""
    global BEST_MODEL, MODEL_METADATA, FEATURE_NAMES

    models_dir = Path(config['training']['save_model_path'])
    model_path = models_dir / "best_model.pkl"
    metadata_path = models_dir / "best_model_metadata.json"

    logger.info(f"Loading best model from: {model_path}")

    # Check if model exists
    if not model_path.exists():
        logger.error(f"Best model not found at: {model_path}")
        logger.error("Please run the training pipeline first!")
        return False

    try:
        # Load model
        BEST_MODEL = joblib.load(model_path)
        logger.info(f"✓ Loaded best model: {type(BEST_MODEL).__name__}")

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                MODEL_METADATA = json.load(f)
            logger.info(f"✓ Model: {MODEL_METADATA['model_name']}")
            logger.info(f"✓ Accuracy: {MODEL_METADATA['all_metrics']['accuracy']:.4f}")
        else:
            logger.warning("Metadata not found, using defaults")
            MODEL_METADATA = {
                'model_name': 'unknown',
                'all_metrics': {'accuracy': 0.0}
            }

        # Load feature names
        try:
            feature_data = pd.read_csv(config['data']['feature_data_path'])
            FEATURE_NAMES = [col for col in feature_data.columns 
                           if col != config['data']['target_column']]
            logger.info(f"✓ Loaded {len(FEATURE_NAMES)} feature names")
        except Exception as e:
            logger.warning(f"Could not load feature names: {str(e)}")

        return True

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


def engineer_features(df):
    """Apply feature engineering to input data"""
    # Drop features that were removed during training
    drop_features = config['features']['drop_features']
    for feat in drop_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])

    # Create engineered features (same as in build_features.py)
    df['SSLDomainTrust'] = df['SSLfinal_State'] * df['Domain_registeration_length']

    df['URLSuspicionScore'] = (
        df['having_IP_Address'] + df['URL_Length'] + 
        df['having_At_Symbol'] + df['Prefix_Suffix']
    ) / 4.0

    df['ContentCredibility'] = (
        df['URL_of_Anchor'] + df['Links_in_tags'] + df['SFH']
    ) / 3.0

    df['DomainReputation'] = (
        df['age_of_domain'] + df['DNSRecord'] + df['web_traffic']
    ) / 3.0

    security_features = ['SSLfinal_State', 'Domain_registeration_length',
                        'HTTPS_token', 'age_of_domain', 'DNSRecord']
    df['SecurityFeaturesCount'] = df[security_features].apply(
        lambda x: (x == 1).sum(), axis=1
    )

    suspicious_features = ['having_IP_Address', 'having_At_Symbol', 'Prefix_Suffix',
                          'having_Sub_Domain', 'Request_URL', 'Abnormal_URL', 'Redirect']
    df['SuspiciousFeaturesCount'] = df[suspicious_features].apply(
        lambda x: (x == -1).sum(), axis=1
    )

    df['SSLAnchorInteraction'] = df['SSLfinal_State'] * df['URL_of_Anchor']

    return df


# ============================================================
# API ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("="*60)
    logger.info("STARTING PHISHING DETECTION API")
    logger.info("="*60)

    success = load_best_model()

    if not success:
        logger.error("Failed to load model. API will not work properly.")
    else:
        logger.info("✓ API ready to serve predictions")
        logger.info("="*60)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Network Security - Phishing Detection API",
        "version": api_config['version'],
        "model": MODEL_METADATA['model_name'] if MODEL_METADATA else "unknown",
        "accuracy": MODEL_METADATA['all_metrics']['accuracy'] if MODEL_METADATA else 0.0,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if BEST_MODEL else "unhealthy",
        version=api_config['version'],
        model_loaded=BEST_MODEL is not None,
        model_name=MODEL_METADATA['model_name'] if MODEL_METADATA else None,
        model_accuracy=MODEL_METADATA['all_metrics']['accuracy'] if MODEL_METADATA else None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if not BEST_MODEL or not MODEL_METADATA:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfoResponse(
        model_name=MODEL_METADATA['model_name'],
        model_type=MODEL_METADATA['model_type'],
        accuracy=MODEL_METADATA['all_metrics']['accuracy'],
        precision=MODEL_METADATA['all_metrics']['precision'],
        recall=MODEL_METADATA['all_metrics']['recall'],
        f1_score=MODEL_METADATA['all_metrics']['f1_score'],
        roc_auc=MODEL_METADATA['all_metrics']['roc_auc'],
        trained_at=MODEL_METADATA['timestamp']
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    if not BEST_MODEL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Convert input to DataFrame
        features_dict = request.features.dict()
        df = pd.DataFrame([features_dict])

        # Apply feature engineering
        df = engineer_features(df)

        # Ensure correct column order
        if FEATURE_NAMES:
            df = df[FEATURE_NAMES]

        # Make prediction
        prediction = BEST_MODEL.predict(df)[0]

        # Get probability if available
        probability = None
        confidence = None
        if hasattr(BEST_MODEL, 'predict_proba'):
            proba = BEST_MODEL.predict_proba(df)[0]
            # Handle XGBoost (predicts 0/1) vs others (predict -1/1)
            if len(proba) == 2:
                probability = float(proba[1])  # Probability of legitimate
                confidence = float(max(proba))

        # Map prediction to label
        prediction_label = "Legitimate" if prediction == 1 else "Phishing"

        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            probability=probability,
            confidence=confidence,
            model_used=MODEL_METADATA['model_name'],
            model_accuracy=MODEL_METADATA['all_metrics']['accuracy'],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    if not BEST_MODEL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert input to DataFrame
        features_list = [f.dict() for f in request.features]
        df = pd.DataFrame(features_list)

        # Apply feature engineering
        df = engineer_features(df)

        # Ensure correct column order
        if FEATURE_NAMES:
            df = df[FEATURE_NAMES]

        # Make predictions
        predictions = BEST_MODEL.predict(df)

        # Get probabilities if available
        probabilities = None
        if hasattr(BEST_MODEL, 'predict_proba'):
            probabilities = BEST_MODEL.predict_proba(df)

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
                    model_used=MODEL_METADATA['model_name'],
                    model_accuracy=MODEL_METADATA['all_metrics']['accuracy'],
                    timestamp=timestamp
                )
            )

        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_predictions=len(predictions),
            model_used=MODEL_METADATA['model_name'],
            timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=api_config['host'],
        port=api_config['port'],
        reload=api_config['reload']
    )
