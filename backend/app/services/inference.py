import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings

logger = logging.getLogger("crop_health_ai.inference")

class ModelLoadError(RuntimeError):
    pass

class InvalidImageError(ValueError):
    pass

class CropHealthInference:
    def __init__(self):
        self.model = None
        self.mapping: Dict[int, str] = {}
        self.config: dict = {}
        self.loaded = False
        self.error: str | None = None

    def load(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

            model_path = Path(settings.model_path)
            mapping_path = Path(settings.mapping_path)
            config_path = Path(settings.config_path)
            if not model_path.exists():
                raise ModelLoadError("The configured model file was not found")

            with mapping_path.open("r", encoding="utf-8") as file:
                raw_mapping = json.load(file)
            self.mapping = {int(index): name for index, name in raw_mapping.items()}
            with config_path.open("r", encoding="utf-8") as file:
                self.config = json.load(file)

            if len(self.mapping) != 38 or set(self.mapping) != set(range(38)):
                raise ModelLoadError("Expected exactly 38 class mappings")

            self.model = tf.keras.models.load_model(str(model_path), compile=False)
            output_shape = self.model.output_shape
            if not (len(output_shape) == 2 and output_shape[-1] == 38):
                raise ModelLoadError(f"Unexpected model output shape: {output_shape}")

            self.preprocess_input = preprocess_input
            self.tf = tf
            self.loaded = True
            logger.info("Loaded MobileNetV2 model: %s", model_path.name)
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Failed to load AI model")

    def preprocess(self, content: bytes) -> Tuple[object, dict]:
        if len(content) > settings.max_upload_bytes:
            raise InvalidImageError("The uploaded image exceeds the 10 MB size limit")
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("The uploaded file is not a valid supported image") from exc

        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise InvalidImageError("Only JPEG, PNG, and WEBP images are supported")

        width, height = image.size
        if width < 32 or height < 32:
            raise InvalidImageError("The image is too small for reliable analysis")

        original_format = image.format
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        raw_array = np.asarray(image, dtype=np.float32)
        red = raw_array[..., 0]
        green = raw_array[..., 1]
        blue = raw_array[..., 2]
        maximum = np.maximum.reduce([red, green, blue])
        minimum = np.minimum.reduce([red, green, blue])
        saturation = np.divide(maximum - minimum, maximum + 1e-6)
        plant_like = (
            ((green >= red * 0.82) & (green >= blue * 0.82) & (saturation >= 0.14))
            | ((green >= red * 0.92) & (red >= 95) & (saturation >= 0.12))
        )
        plant_like_score = float(np.clip(np.mean(plant_like), 0.0, 1.0))
        color_std = float(np.mean(np.std(raw_array, axis=(0, 1))))
        edge_strength = float(
            np.mean(np.abs(np.diff(raw_array, axis=0))) + np.mean(np.abs(np.diff(raw_array, axis=1)))
        )

        array = self.preprocess_input(raw_array)
        tensor = self.tf.convert_to_tensor(array[None, ...])
        info = {
            "filename_format": original_format or "image",
            "width": width,
            "height": height,
            "processed_size": [224, 224],
            "plant_like_score": plant_like_score,
            "color_std": round(color_std, 3),
            "edge_strength": round(edge_strength, 3),
        }
        return tensor, info

    def _probabilities(self, scores: np.ndarray) -> np.ndarray:
        output_layer = self.model.layers[-1]
        activation = getattr(getattr(output_layer, "activation", None), "__name__", "")
        if activation == "softmax":
            probabilities = scores
            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.isclose(np.sum(probabilities), 1.0, atol=1e-4)
            ):
                raise ModelLoadError("Model softmax output was not a valid probability distribution")
            return probabilities

        shifted_scores = scores - np.max(scores)
        exp_scores = np.exp(shifted_scores)
        return exp_scores / np.sum(exp_scores)

    def assess_domain(self, top: list, image_info: dict) -> tuple[str, str | None]:
        plant_like_score = float(image_info.get("plant_like_score", 0.0))
        color_std = float(image_info.get("color_std", 0.0))
        edge_strength = float(image_info.get("edge_strength", 0.0))
        confidence = float(top[0]["confidence"])
        margin = confidence - float(top[1]["confidence"]) if len(top) > 1 else confidence

        clearly_not_leaf = (
            plant_like_score < 0.08
            and color_std < 35.0
            and edge_strength < 18.0
        ) or (plant_like_score < 0.025 and color_std < 18.0)
        if clearly_not_leaf:
            return (
                "outside_domain",
                "This image may not belong to the 38 categories supported by Crop Health AI. Please upload a clear image of a supported plant leaf.",
            )

        if plant_like_score < 0.12 or confidence < settings.low_confidence_threshold or margin < 0.04:
            return (
                "uncertain",
                "This result is uncertain. The image may be unclear or outside the 38 supported categories; upload a clear, close-up image of a supported plant leaf or consult an agricultural specialist.",
            )

        return "supported_prediction", None

    def predict(self, content: bytes, top_k: int = 5):
        if not self.loaded:
            raise ModelLoadError("AI model is not loaded")
        tensor, image_info = self.preprocess(content)
        scores = np.asarray(self.model(tensor, training=False).numpy()[0], dtype=np.float32)
        if scores.shape[0] != 38:
            raise ModelLoadError("Model returned an unexpected number of class scores")
        if not np.all(np.isfinite(scores)):
            raise ModelLoadError("Model returned non-finite class scores")
        probabilities = self._probabilities(scores)
        indices = np.argsort(probabilities)[::-1][:top_k]
        top = [{"class_id": int(index), "class_name": self.mapping[int(index)], "confidence": float(probabilities[index])} for index in indices]
        return top, image_info

inference_service = CropHealthInference()
