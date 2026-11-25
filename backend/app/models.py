from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.database import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(String, primary_key=True, index=True)
    predicted_class_id = Column(Integer, nullable=False)
    predicted_class_name = Column(String, nullable=False)
    crop_name = Column(String, nullable=False)
    disease_name = Column(String, nullable=False)
    is_healthy = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    alternatives_json = Column(Text, nullable=False, default="[]")
    recommendation_json = Column(Text, nullable=False, default="{}")
    image_filename = Column(String, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    domain_status = Column(String, nullable=False, default="uncertain")
    plant_like_score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
