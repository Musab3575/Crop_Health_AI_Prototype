import json
from pathlib import Path

CLASS_NAMES = json.loads(Path(__file__).resolve().parents[3] / "ml_assets" / "metadata" / "class_mapping.json") if False else None

GENERAL_DISEASE = {
    "description": "A leaf abnormality was detected. The result is AI-assisted guidance and may require local expert confirmation.",
    "symptoms": ["Leaf spots, discoloration, wilting, lesions, or abnormal growth", "Changes may differ by crop and disease stage"],
    "possible_factors": ["Fungal or bacterial infection", "Pests, nutrient stress, weather, or environmental damage"],
    "prevention": ["Use clean planting material and inspect crops regularly", "Improve spacing and airflow; avoid unnecessary overhead watering", "Remove seriously affected plant material according to local guidance"],
    "management": ["Isolate and monitor affected plants", "Seek local agricultural extension advice before choosing any treatment", "If chemicals are considered, follow the registered product label and local regulations"],
    "monitoring": ["Check nearby plants and document new symptoms", "Photograph affected leaves in clear, natural light over several days"],
    "seek_expert_when": ["Symptoms spread rapidly or affect much of the crop", "You are unsure of the diagnosis or the issue recurred"],
}

HEALTHY = {
    "description": "The model predicts this leaf appears healthy. Continue routine observation because no image-based test can rule out every problem.",
    "symptoms": ["No strongly abnormal disease pattern was predicted"],
    "possible_factors": ["Healthy appearance depends on image quality and the model's supported crop classes"],
    "prevention": ["Maintain balanced nutrition and appropriate irrigation", "Keep tools and planting areas clean", "Rotate crops where practical"],
    "management": ["Continue routine care and monitor new growth"],
    "monitoring": ["Inspect regularly for new spots, discoloration, or pests"],
    "seek_expert_when": ["Visible symptoms persist or spread despite normal-looking initial results"]
}

SPECIAL = {
    "Haunglongbing_(Citrus_greening)": "Citrus greening is a serious disease spread by insects. Restrict movement of infected material and seek official or local extension support promptly.",
    "Early_blight": "Early blight can produce target-like leaf spots. Remove severely affected leaves and avoid prolonged leaf wetness where practical.",
    "Late_blight": "Late blight can spread quickly in cool, wet conditions. Monitor the whole plant and seek local advice promptly.",
    "Powdery_mildew": "Powdery mildew commonly appears as a white or pale powdery coating. Improve airflow and monitor repeated outbreaks.",
    "Bacterial_spot": "Bacterial spots may look different across crops. Avoid moving contaminated tools or plant debris and seek local guidance.",
    "Apple_scab": "Apple scab commonly causes olive-brown lesions on leaves. Remove fallen diseased material and prune for airflow where appropriate.",
    "Black_rot": "Black rot may cause leaf lesions and fruit decline. Remove affected plant material and follow crop-specific local guidance.",
}

class RecommendationService:
    def get(self, crop_name: str, disease_name: str, is_healthy: bool, low_confidence: bool = False):
        if is_healthy:
            base = HEALTHY
            status = "Healthy"
        else:
            base = GENERAL_DISEASE
            status = "Disease or abnormal condition"
            for key, detail in SPECIAL.items():
                if key in disease_name:
                    base = {**base, "description": detail}
                    break
        if low_confidence:
            management = list(base["management"])
            management.insert(0, "Use caution: the model confidence is relatively low. Capture a clearer leaf image and consider expert verification.")
        else:
            management = base["management"]
        return {
            "crop": crop_name,
            "status": status,
            "description": base["description"],
            "symptoms": base["symptoms"],
            "possible_factors": base["possible_factors"],
            "prevention": base["prevention"],
            "management": management,
            "monitoring": base["monitoring"],
            "seek_expert_when": base["seek_expert_when"],
        }

recommendation_service = RecommendationService()
