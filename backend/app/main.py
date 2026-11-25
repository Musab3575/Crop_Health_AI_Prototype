import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import Base, engine, get_db
from app.models import PredictionRecord
from app.schemas import AlternativePrediction, HistoryPage, PredictionResponse, StatsResponse, SupportedClass
from app.services.inference import InvalidImageError, ModelLoadError, inference_service
from app.services.recommendations import recommendation_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("crop_health_ai")

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        existing = {row[1] for row in connection.execute(text("PRAGMA table_info(prediction_history)")).fetchall()}
        if "domain_status" not in existing:
            connection.execute(text("ALTER TABLE prediction_history ADD COLUMN domain_status VARCHAR NOT NULL DEFAULT 'uncertain'"))
        if "plant_like_score" not in existing:
            connection.execute(text("ALTER TABLE prediction_history ADD COLUMN plant_like_score FLOAT NOT NULL DEFAULT 0"))
    inference_service.load()

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "model_status": "loaded" if inference_service.loaded else "unavailable", "app_version": settings.app_version, "model_loaded": inference_service.loaded}


def parse_class(class_name: str):
    parts = class_name.split("___", 1)
    crop = parts[0].replace("_", " ").replace("(including_sour)", "(including sour)")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = disease.lower() == "healthy"
    return crop, disease, is_healthy

@app.get("/api/v1/classes", response_model=list[SupportedClass])
def supported_classes():
    return [SupportedClass(class_id=i, class_name=name, crop_name=parse_class(name)[0], disease_name=parse_class(name)[1], is_healthy=parse_class(name)[2]) for i, name in sorted(inference_service.mapping.items())]

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WEBP images are supported")
    content = await file.read(settings.max_upload_bytes + 1)
    try:
        top, image_info = inference_service.predict(content)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail="The AI model is unavailable")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="The image could not be analyzed")

    first = top[0]
    crop, disease, healthy = parse_class(first["class_name"])
    low = first["confidence"] < settings.low_confidence_threshold
    domain_status, domain_warning = inference_service.assess_domain(top, image_info)
    alternatives = []
    for item in top:
        acrop, adisease, _ = parse_class(item["class_name"])
        alternatives.append(AlternativePrediction(class_id=item["class_id"], class_name=item["class_name"], crop_name=acrop, disease_name=adisease, confidence=item["confidence"]))
    recommendation = recommendation_service.get(crop, disease, healthy, low)
    prediction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    record = PredictionRecord(
        id=prediction_id, predicted_class_id=first["class_id"], predicted_class_name=first["class_name"], crop_name=crop,
        disease_name=disease, is_healthy=healthy, confidence=first["confidence"], alternatives_json=json.dumps([a.model_dump() for a in alternatives]),
        recommendation_json=json.dumps(recommendation), image_filename=None, image_width=image_info["width"], image_height=image_info["height"],
        domain_status=domain_status, plant_like_score=float(image_info.get("plant_like_score", 0.0)), created_at=timestamp,
    )
    db.add(record)
    db.commit()

    explanation = ("The model class predicts that the leaf is healthy." if healthy else "The model detected a possible disease pattern associated with this crop leaf.") + " This is AI-assisted guidance, not a confirmed diagnosis."
    warning = "AI confidence is relatively low for this image. Try taking another clear image of the leaf or consult an agricultural specialist." if low else None
    return PredictionResponse(prediction_id=prediction_id, predicted_class_id=first["class_id"], predicted_class_name=first["class_name"], crop_name=crop,
        disease_name=disease, is_healthy=healthy, confidence=first["confidence"], top_predictions=alternatives, recommendation=recommendation,
        explanation=explanation, low_confidence_warning=warning,
        domain_status=domain_status, domain_warning=domain_warning,
        plant_like_score=float(image_info.get("plant_like_score", 0.0)), image_info=image_info, timestamp=timestamp)

@app.get("/api/v1/history", response_model=HistoryPage)
def history(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), search: Optional[str] = None,
           status: Optional[str] = None, sort: str = "newest", db: Session = Depends(get_db)):
    query = db.query(PredictionRecord)
    if search:
        term = f"%{search}%"
        query = query.filter((PredictionRecord.predicted_class_name.ilike(term)) | (PredictionRecord.crop_name.ilike(term)) | (PredictionRecord.disease_name.ilike(term)))
    if status == "healthy":
        query = query.filter(PredictionRecord.is_healthy.is_(True))
    elif status == "disease":
        query = query.filter(PredictionRecord.is_healthy.is_(False))
    total = query.count()
    query = query.order_by(PredictionRecord.created_at.desc() if sort != "oldest" else PredictionRecord.created_at.asc())
    records = query.offset((page - 1) * page_size).limit(page_size).all()
    items = [_record_response(record) for record in records]
    return HistoryPage(items=items, total=total, page=page, page_size=page_size, pages=math.ceil(total / page_size) if total else 0)

def _record_response(record: PredictionRecord):
    alternatives = json.loads(record.alternatives_json)
    recommendation = json.loads(record.recommendation_json)
    low = record.confidence < settings.low_confidence_threshold
    timestamp = record.created_at if record.created_at.tzinfo else record.created_at.replace(tzinfo=timezone.utc)
    return PredictionResponse(prediction_id=record.id, predicted_class_id=record.predicted_class_id, predicted_class_name=record.predicted_class_name,
        crop_name=record.crop_name, disease_name=record.disease_name, is_healthy=record.is_healthy, confidence=record.confidence,
        top_predictions=[AlternativePrediction(**item) for item in alternatives], recommendation=recommendation,
        explanation=("The model class predicts that the leaf is healthy." if record.is_healthy else "The model detected a possible disease pattern associated with this crop leaf.") + " This is AI-assisted guidance, not a confirmed diagnosis.",
        low_confidence_warning=("AI confidence is relatively low for this image. Try taking another clear image of the leaf or consult an agricultural specialist." if low else None),
        domain_status=getattr(record, "domain_status", "uncertain"),
        domain_warning=("This image may not belong to the 38 categories supported by Crop Health AI. Please upload a clear image of a supported plant leaf." if getattr(record, "domain_status", "uncertain") == "outside_domain" else None),
        plant_like_score=float(getattr(record, "plant_like_score", 0.0)),
        image_info={"width": record.image_width, "height": record.image_height, "processed_size": [224, 224]}, timestamp=timestamp)

@app.get("/api/v1/history/{prediction_id}", response_model=PredictionResponse)
def history_detail(prediction_id: str, db: Session = Depends(get_db)):
    record = db.get(PredictionRecord, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan record not found")
    return _record_response(record)

@app.delete("/api/v1/history/{prediction_id}")
def delete_history(prediction_id: str, db: Session = Depends(get_db)):
    record = db.get(PredictionRecord, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan record not found")
    db.delete(record); db.commit()
    return {"deleted": True, "prediction_id": prediction_id}

@app.delete("/api/v1/history")
def delete_all_history(db: Session = Depends(get_db)):
    count = db.query(PredictionRecord).count()
    db.query(PredictionRecord).delete(); db.commit()
    return {"deleted": True, "deleted_count": count}

@app.get("/api/v1/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)):
    total = db.query(PredictionRecord).count()
    healthy = db.query(PredictionRecord).filter(PredictionRecord.is_healthy.is_(True)).count()
    low = db.query(PredictionRecord).filter(PredictionRecord.confidence < settings.low_confidence_threshold).count()
    most = db.query(PredictionRecord.predicted_class_name, func.count(PredictionRecord.id).label("count")).group_by(PredictionRecord.predicted_class_name).order_by(func.count(PredictionRecord.id).desc()).first()
    records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(7).all()
    activity = [{"prediction_id": r.id, "crop_name": r.crop_name, "disease_name": r.disease_name, "is_healthy": r.is_healthy, "confidence": r.confidence, "created_at": r.created_at.isoformat()} for r in records]
    return StatsResponse(total_scans=total, healthy_detections=healthy, disease_detections=total - healthy, low_confidence_detections=low, most_detected=most[0] if most else None, recent_activity=activity)
