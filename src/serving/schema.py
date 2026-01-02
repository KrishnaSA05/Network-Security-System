"""
API Schema Definitions
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime

class PhishingFeatures(BaseModel):
    """Input features for phishing prediction"""
    having_IP_Address: int = Field(..., ge=-1, le=1, description="IP Address in URL (-1, 1)")
    URL_Length: int = Field(..., ge=-1, le=1, description="URL Length (-1, 0, 1)")
    Shortining_Service: int = Field(..., ge=-1, le=1, description="Shortening Service (-1, 1)")
    having_At_Symbol: int = Field(..., ge=-1, le=1, description="@ Symbol in URL (-1, 1)")
    double_slash_redirecting: int = Field(..., ge=-1, le=1, description="Double Slash Redirect (-1, 1)")
    Prefix_Suffix: int = Field(..., ge=-1, le=1, description="Prefix/Suffix (-1, 1)")
    having_Sub_Domain: int = Field(..., ge=-1, le=1, description="Subdomain (-1, 0, 1)")
    SSLfinal_State: int = Field(..., ge=-1, le=1, description="SSL State (-1, 0, 1)")
    Domain_registeration_length: int = Field(..., ge=-1, le=1, description="Domain Registration Length (-1, 1)")
    Favicon: int = Field(..., ge=-1, le=1, description="Favicon (-1, 1)")
    port: int = Field(..., ge=-1, le=1, description="Port (-1, 1)")
    HTTPS_token: int = Field(..., ge=-1, le=1, description="HTTPS Token (-1, 1)")
    Request_URL: int = Field(..., ge=-1, le=1, description="Request URL (-1, 1)")
    URL_of_Anchor: int = Field(..., ge=-1, le=1, description="URL of Anchor (-1, 0, 1)")
    Links_in_tags: int = Field(..., ge=-1, le=1, description="Links in Tags (-1, 0, 1)")
    SFH: int = Field(..., ge=-1, le=1, description="SFH (-1, 0, 1)")
    Submitting_to_email: int = Field(..., ge=-1, le=1, description="Submit to Email (-1, 1)")
    Abnormal_URL: int = Field(..., ge=-1, le=1, description="Abnormal URL (-1, 1)")
    Redirect: int = Field(..., ge=-1, le=1, description="Redirect (0, 1)")
    on_mouseover: int = Field(..., ge=-1, le=1, description="onMouseOver (-1, 1)")
    RightClick: int = Field(..., ge=-1, le=1, description="Right Click (-1, 1)")
    popUpWidnow: int = Field(..., ge=-1, le=1, description="PopUp Window (-1, 1)")
    Iframe: int = Field(..., ge=-1, le=1, description="IFrame (-1, 1)")
    age_of_domain: int = Field(..., ge=-1, le=1, description="Age of Domain (-1, 1)")
    DNSRecord: int = Field(..., ge=-1, le=1, description="DNS Record (-1, 1)")
    web_traffic: int = Field(..., ge=-1, le=1, description="Web Traffic (-1, 0, 1)")
    Page_Rank: int = Field(..., ge=-1, le=1, description="Page Rank (-1, 1)")
    Google_Index: int = Field(..., ge=-1, le=1, description="Google Index (-1, 1)")
    Links_pointing_to_page: int = Field(..., ge=-1, le=1, description="Links Pointing to Page (-1, 0, 1)")
    Statistical_report: int = Field(..., ge=-1, le=1, description="Statistical Report (-1, 1)")
    
    class Config:
        schema_extra = {
            "example": {
                "having_IP_Address": -1,
                "URL_Length": 1,
                "Shortining_Service": 1,
                "having_At_Symbol": 1,
                "double_slash_redirecting": -1,
                "Prefix_Suffix": -1,
                "having_Sub_Domain": -1,
                "SSLfinal_State": -1,
                "Domain_registeration_length": -1,
                "Favicon": 1,
                "port": 1,
                "HTTPS_token": -1,
                "Request_URL": 1,
                "URL_of_Anchor": 0,
                "Links_in_tags": 1,
                "SFH": -1,
                "Submitting_to_email": -1,
                "Abnormal_URL": -1,
                "Redirect": 0,
                "on_mouseover": 1,
                "RightClick": 1,
                "popUpWidnow": 1,
                "Iframe": 1,
                "age_of_domain": -1,
                "DNSRecord": -1,
                "web_traffic": -1,
                "Page_Rank": -1,
                "Google_Index": 1,
                "Links_pointing_to_page": 1,
                "Statistical_report": -1
            }
        }

class PredictionRequest(BaseModel):
    """Single prediction request"""
    features: PhishingFeatures
    model_name: Optional[str] = Field("random_forest", description="Model to use for prediction")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    features: List[PhishingFeatures]
    model_name: Optional[str] = Field("random_forest", description="Model to use for prediction")

class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: int = Field(..., description="Prediction: -1 (phishing) or 1 (legitimate)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability: Optional[float] = Field(None, description="Prediction probability (if available)")
    confidence: Optional[float] = Field(None, description="Confidence score")
    model_used: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_predictions: int
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: List[str]
    timestamp: str

class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    model_type: str
    version: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    last_trained: Optional[str] = None

class ModelsListResponse(BaseModel):
    """List of available models"""
    models: List[ModelInfo]
    total_models: int
    default_model: str
