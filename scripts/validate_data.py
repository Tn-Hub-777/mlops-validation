"""Standalone data validation entry point."""

from src.data import load_dataset

if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset validation passed: {len(df)} rows, {len(df.columns)} columns.")
