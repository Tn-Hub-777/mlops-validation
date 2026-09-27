"""Train baseline and candidate model, evaluate the quality gate, and save artifacts."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import EXPECTED_FEATURES, TARGET, load_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.20
MARGIN = 0.05


def build_candidate() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_dataset()

        X = df[EXPECTED_FEATURES]
        y = df[TARGET]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_score = balanced_accuracy_score(
            y_valid, baseline.predict(X_valid)
        )

        candidate = build_candidate()
        candidate.fit(X_train.iloc[:1], y_train.iloc[:1])
        model_score = balanced_accuracy_score(
            y_valid, candidate.predict(X_valid)
        )

        gate_passed = model_score >= baseline_score + MARGIN

        metrics = {
            "dataset": "UCI Banknote Authentication",
            "task": "Binary classification",
            "metric": "balanced_accuracy",
            "higher_is_better": True,
            "random_state": RANDOM_STATE,
            "validation_fraction": TEST_SIZE,
            "baseline_score": round(float(baseline_score), 6),
            "model_score": round(float(model_score), 6),
            "margin": MARGIN,
            "required_score": round(float(baseline_score + MARGIN), 6),
            "gate_result": "PASS" if gate_passed else "FAIL",
        }

        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        if not gate_passed:
            print(
                f"QUALITY GATE FAILED: model={model_score:.4f}, "
                f"baseline={baseline_score:.4f}, margin={MARGIN:.4f}"
            )
            return 1

        joblib.dump(candidate, output_dir / "model.joblib")
        print(
            f"QUALITY GATE PASSED: model={model_score:.4f} >= "
            f"{baseline_score:.4f} + {MARGIN:.4f}"
        )
        return 0

    except Exception as exc:
        print(f"TRAINING/VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
