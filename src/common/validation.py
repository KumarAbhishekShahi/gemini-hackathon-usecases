"""Validation helpers used before sending cleaned records to Gemini."""

from pathlib import Path
import json
import pandas as pd

def require_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")

def require_min_rows(df: pd.DataFrame, minimum_rows: int, dataset_name: str) -> None:
    if len(df) < minimum_rows:
        raise ValueError(f"{dataset_name} has only {len(df)} rows. Expected at least {minimum_rows}.")

def write_validation_report(report_path: Path, payload: dict) -> None:
    report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
