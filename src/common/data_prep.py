"""Shared cleaning helpers for all Gemini hackathon use cases."""

import re
import pandas as pd

def normalize_whitespace(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()

def normalize_case(text: str) -> str:
    text = normalize_whitespace(text)
    if text.isupper() and len(text) > 8:
        text = text.title()
    return text

def safe_fill(series: pd.Series, default_value: str) -> pd.Series:
    return series.fillna(default_value).replace('', default_value)

def deduplicate_rows(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
