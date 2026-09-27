"""Prediction application."""

from pathlib import Path

import joblib
import pandas as pd

from src.data import EXPECTED_FEATURES

DEFAULT_MODEL = Path("artifacts/model.joblib")


def predict(sample: dict, model_path: str | Path = DEFAULT_MODEL):
    missing = [c for c in EXPECTED_FEATURES if c not in sample]
    if missing:
        raise ValueError(
            f"Missing required feature(s): {', '.join(missing)}"
        )

    extra = [c for c in sample if c not in EXPECTED_FEATURES]
    if extra:
        raise ValueError(f"Unexpected feature(s): {', '.join(extra)}")

    row = pd.DataFrame(
        [[sample[c] for c in EXPECTED_FEATURES]],
        columns=EXPECTED_FEATURES,
    )

    model = joblib.load(model_path)
    prediction = model.predict(row)

    if prediction.shape != (1,):
        raise RuntimeError(
            f"Unexpected prediction shape: {prediction.shape}"
        )

    return prediction[0]


if __name__ == "__main__":
    example = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699,
    }
    print(predict(example))
