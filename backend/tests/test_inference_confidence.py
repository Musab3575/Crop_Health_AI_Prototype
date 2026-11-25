from types import SimpleNamespace

import numpy as np

from app.services.inference import CropHealthInference


def test_softmax_outputs_are_not_softmaxed_a_second_time():
    service = CropHealthInference()
    service.model = SimpleNamespace(
        layers=[SimpleNamespace(activation=SimpleNamespace(__name__="softmax"))]
    )
    model_probabilities = np.array([0.72, 0.18, 0.10], dtype=np.float32)

    probabilities = service._probabilities(model_probabilities)

    np.testing.assert_array_equal(probabilities, model_probabilities)
    assert float(probabilities.max()) == float(model_probabilities.max())


def test_logits_are_converted_to_probabilities():
    service = CropHealthInference()
    service.model = SimpleNamespace(
        layers=[SimpleNamespace(activation=SimpleNamespace(__name__="linear"))]
    )
    logits = np.array([3.0, 1.0, -2.0], dtype=np.float32)

    probabilities = service._probabilities(logits)

    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities.argmax() == logits.argmax()
    assert probabilities[0] > probabilities[1] > probabilities[2]
