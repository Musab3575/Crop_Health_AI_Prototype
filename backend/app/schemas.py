from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class AlternativePrediction(BaseModel):
    class_id: int
    class_name: str
    crop_name: str
    disease_name: str
    confidence: float

class Recommendation(BaseModel):
    crop: str
    status: str
    description: str
    symptoms: List[str]
    possible_factors: List[str]
    prevention: List[str]
    management: List[str]
    monitoring: List[str]
    seek_expert_when: List[str]

class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_class_id: int
    predicted_class_name: str
    crop_name: str
    disease_name: str
    is_healthy: bool
    confidence: float
    top_predictions: List[AlternativePrediction]
    recommendation: Recommendation
    explanation: str
    low_confidence_warning: Optional[str] = None
    domain_status: str = "uncertain"
    domain_warning: Optional[str] = None
    plant_like_score: float = 0.0
    image_info: dict
    timestamp: datetime

class SupportedClass(BaseModel):
    class_id: int
    class_name: str
    crop_name: str
    disease_name: str
    is_healthy: bool

class HistoryPage(BaseModel):
    items: List[PredictionResponse]
    total: int
    page: int
    page_size: int
    pages: int

class StatsResponse(BaseModel):
    total_scans: int
    healthy_detections: int
    disease_detections: int
    low_confidence_detections: int
    most_detected: Optional[str] = None
    recent_activity: List[dict]
