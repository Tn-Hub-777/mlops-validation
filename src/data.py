"""Dataset download and schema validation."""

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests

DATA_URL = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"
EXPECTED_FEATURES = ["variance", "skewness", "curtosis", "entropy"]
TARGET = "class"


def load_dataset() -> pd.DataFrame:
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        names = archive.namelist()
        csv_name = next(
            (name for name in names if name.lower().endswith((".csv", ".txt"))),
            None,
        )
        if csv_name is None:
            raise ValueError("Dataset archive does not contain a CSV file.")

        with archive.open(csv_name) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=EXPECTED_FEATURES + [TARGET],
            )

    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing_features = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns: {missing_features}"
        )

    if TARGET not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET}")

    if df[EXPECTED_FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if len(df) < 100:
        raise ValueError("Dataset is unexpectedly small.")
