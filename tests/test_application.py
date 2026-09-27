"""Application tests. Training is executed first in the same CI job."""

from pathlib import Path

import joblib
import pytest

from src.predict import predict

MODEL_PATH = Path("artifacts/model.joblib")

VALID_SAMPLE = {
    "variance": 3.6216,
    "skewness": 8.6661,
    "curtosis": -2.8073,
    "entropy": -0.44699,
}


def test_saved_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")


def test_valid_sample_returns_expected_shape_and_type():
    model = joblib.load(MODEL_PATH)
    result = model.predict(
        [[
            VALID_SAMPLE["variance"],
            VALID_SAMPLE["skewness"],
            VALID_SAMPLE["curtosis"],
            VALID_SAMPLE["entropy"],
        ]]
    )
    assert result.shape == (1,)
    assert hasattr(result[0], "item") or isinstance(result[0], (int, float))


def test_prediction_function_accepts_valid_sample():
    result = predict(VALID_SAMPLE, MODEL_PATH)
    assert result is not None


def test_missing_required_feature_is_rejected():
    bad_sample = dict(VALID_SAMPLE)
    del bad_sample["entropy"]

    with pytest.raises(ValueError, match="Missing required feature"):
        predict(bad_sample, MODEL_PATH)
