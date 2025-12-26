import os
import yaml
import pandas as pd
from scipy.stats import ks_2samp


def load_schema(schema_path: str):
    with open(schema_path) as f:
        return yaml.safe_load(f)


def validate_columns(df: pd.DataFrame, schema: dict) -> bool:
    return len(df.columns) == len(schema["columns"])


def detect_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, report_path: str, threshold=0.05):
    drift_report = {}
    status = True

    for col in train_df.columns:
        ks_result = ks_2samp(train_df[col], test_df[col])
        drift = ks_result.pvalue < threshold
        drift_report[col] = {
            "p_value": float(ks_result.pvalue),
            "drift": drift
        }
        if drift:
            status = False

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        yaml.safe_dump(drift_report, f)

    return status


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    schema = load_schema("src/config/schema.yaml")

    if not validate_columns(train, schema):
        raise ValueError("Train data schema mismatch")

    if not validate_columns(test, schema):
        raise ValueError("Test data schema mismatch")

    detect_drift(train, test, "data/validation/drift_report.yaml")
