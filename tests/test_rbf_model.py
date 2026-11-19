import json
import math

import pytest

from n16_eeg.model import load_model


def test_rbf_model_loads_and_applies_training_transform(tmp_path):
    payload = {
        "model_type": "numpy_rbf_svm",
        "format_version": 1,
        "feature_names": ["theta_alpha"],
        "center": [0.2],
        "scale": [0.5],
        "support_vectors": [[0.0], [1.0]],
        "dual_coefficients": [-0.6, 0.6],
        "intercept": 0.1,
        "gamma": 2.0,
        "threshold": 0.57,
    }
    path = tmp_path / "rbf.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    model = load_model(path)
    expected_margin = -0.6 * math.exp(-2.0) + 0.6 + 0.1

    assert model.threshold == 0.57
    assert model.predict_feature_dict({"theta_alpha": math.expm1(0.7)}) == pytest.approx(
        1.0 / (1.0 + math.exp(-expected_margin))
    )

    payload["scale"] = [0.0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid RBF model parameters"):
        load_model(path)
