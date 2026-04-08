"""Prepare messy raw CSV files into clean datasets for the Gemini demos."""

from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.data_prep import normalize_case, normalize_whitespace, safe_fill, deduplicate_rows
from src.common.validation import require_columns, require_min_rows, write_validation_report


def clean_tickets(raw_path: Path, out_path: Path) -> dict:
    print("\n[Ticket Prep] Loading raw support tickets...")
    df = pd.read_csv(raw_path)
    require_columns(df, ['ticket_id', 'subject', 'body', 'source', 'app'], 'tickets_raw')
    require_min_rows(df, 1000, 'tickets_raw')
    print(f"[Ticket Prep] Raw rows loaded: {len(df)}")
    df['subject'] = df['subject'].map(normalize_case)
    df['body'] = df['body'].map(normalize_whitespace)
    df['customer_tier'] = safe_fill(df['customer_tier'], 'unknown')
    df['priority_hint'] = safe_fill(df['priority_hint'], 'normal')
    df['reporter_email'] = safe_fill(df['reporter_email'], 'unknown@example.com')
    df = deduplicate_rows(df, ['ticket_id'])
    df.to_csv(out_path, index=False)
    print(f"[Ticket Prep] Clean rows written: {len(df)} -> {out_path}")
    return {'dataset': 'tickets', 'rows': int(len(df)), 'output': str(out_path)}


def clean_requirements(raw_path: Path, out_path: Path) -> dict:
    print("\n[Requirements Prep] Loading raw requirements...")
    df = pd.read_csv(raw_path)
    require_columns(df, ['requirement_id', 'raw_requirement_text', 'product_area'], 'requirements_raw')
    require_min_rows(df, 1000, 'requirements_raw')
    print(f"[Requirements Prep] Raw rows loaded: {len(df)}")
    for column in ['raw_requirement_text', 'business_context', 'product_area', 'requester_role']:
        df[column] = df[column].map(normalize_case)
    df['priority_hint'] = safe_fill(df['priority_hint'], 'medium')
    df = deduplicate_rows(df, ['requirement_id'])
    df.to_csv(out_path, index=False)
    print(f"[Requirements Prep] Clean rows written: {len(df)} -> {out_path}")
    return {'dataset': 'requirements', 'rows': int(len(df)), 'output': str(out_path)}


def clean_logs(raw_path: Path, out_path: Path) -> dict:
    print("\n[Log Prep] Loading raw incident logs...")
    df = pd.read_csv(raw_path)
    require_columns(df, ['incident_id', 'log_chunk', 'service_name'], 'logs_raw')
    require_min_rows(df, 1000, 'logs_raw')
    print(f"[Log Prep] Raw rows loaded: {len(df)}")
    df['log_chunk'] = df['log_chunk'].map(normalize_whitespace)
    df['incident_title'] = df['incident_title'].map(normalize_case)
    df = deduplicate_rows(df, ['incident_id'])
    df.to_csv(out_path, index=False)
    print(f"[Log Prep] Clean rows written: {len(df)} -> {out_path}")
    return {'dataset': 'logs', 'rows': int(len(df)), 'output': str(out_path)}


def main() -> None:
    raw_dir = PROJECT_ROOT / 'data' / 'raw'
    clean_dir = PROJECT_ROOT / 'data' / 'clean'
    validation_report_path = PROJECT_ROOT / 'docs' / 'dataset_validation_report.json'
    print('=' * 80)
    print('STARTING DATASET PREPARATION')
    print('=' * 80)
    summary = {
        'tickets': clean_tickets(raw_dir / 'tickets_raw.csv', clean_dir / 'tickets_clean.csv'),
        'requirements': clean_requirements(raw_dir / 'requirements_raw.csv', clean_dir / 'requirements_clean.csv'),
        'logs': clean_logs(raw_dir / 'logs_raw.csv', clean_dir / 'logs_clean.csv'),
    }
    write_validation_report(validation_report_path, summary)
    print('\nValidation summary written to:', validation_report_path)
    print('Dataset preparation complete.')


if __name__ == '__main__':
    main()
