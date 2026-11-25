from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    app_name: str = "Crop Health AI"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    database_url: str = f"sqlite:///{ROOT / 'storage' / 'crop_health.db'}"
    model_path: str = str(ROOT / "ml_assets" / "model" / "mobilenetv2_stage2_best.keras")
    mapping_path: str = str(ROOT / "ml_assets" / "metadata" / "class_mapping.json")
    config_path: str = str(ROOT / "ml_assets" / "metadata" / "model_config.json")
    max_upload_bytes: int = 10 * 1024 * 1024
    low_confidence_threshold: float = 0.40

    model_config = SettingsConfigDict(env_file=ROOT / "backend" / ".env", extra="ignore")

settings = Settings()
