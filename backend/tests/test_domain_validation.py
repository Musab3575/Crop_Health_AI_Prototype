from app.services.inference import CropHealthInference


def test_clear_low_variance_image_is_outside_domain():
    service = CropHealthInference()
    top = [{"confidence": 0.42}, {"confidence": 0.12}]
    status, warning = service.assess_domain(top, {
        "plant_like_score": 0.02,
        "color_std": 4.0,
        "edge_strength": 2.0,
    })
    assert status == "outside_domain"
    assert "38 categories" in (warning or "")


def test_low_confidence_leaf_like_image_is_uncertain():
    service = CropHealthInference()
    top = [{"confidence": 0.18}, {"confidence": 0.14}]
    status, warning = service.assess_domain(top, {
        "plant_like_score": 0.45,
        "color_std": 70.0,
        "edge_strength": 25.0,
    })
    assert status == "uncertain"
    assert warning


def test_confident_leaf_like_prediction_is_supported():
    service = CropHealthInference()
    top = [{"confidence": 0.92}, {"confidence": 0.03}]
    status, warning = service.assess_domain(top, {
        "plant_like_score": 0.55,
        "color_std": 68.0,
        "edge_strength": 30.0,
    })
    assert status == "supported_prediction"
    assert warning is None
