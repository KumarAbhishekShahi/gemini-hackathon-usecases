"""
log_explainer.py
================
USE CASE 3: Incident Log Explainer using Gemini Structured Output.

Calls Gemini via the shared REST client (src/common/gemini_client.py).
No gRPC. No google-genai SDK. Pure HTTP via 'requests'.

Run:
  python src/use_cases/log_explainer.py
  python src/use_cases/log_explainer.py --env prod --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.gemini_client import call_gemini_rest as _rest_call
from src.common.data_prep import normalize_whitespace, safe_fill
from src.common.validation import require_columns, require_min_rows

GEMINI_MODEL    = "gemini-2.5-flash"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "logs_clean.csv"
OUTPUT_DIR      = PROJECT_ROOT / "output" / "log_explainer"
REQUIRED_COLUMNS = [
    "incident_id", "log_chunk", "service_name", "environment", "region",
]


# ── Pydantic schema ────────────────────────────────────────────────────────

class IncidentAnalysis(BaseModel):
    affected_component: str
    probable_root_cause: str
    confidence: int = Field(ge=1, le=100)
    what_happened: str
    evidence_lines: list[str]
    immediate_actions: list[str] = Field(min_length=1)
    code_fix_suggestions: list[str]
    monitoring_checks: list[str]
    incident_severity: str = Field(pattern=r"^(sev1|sev2|sev3|sev4)$")


# ── Input / output validation ──────────────────────────────────────────────

def validate_row(row: pd.Series) -> tuple[bool, list[str]]:
    issues = []
    if len(str(row.get("log_chunk", "")).strip()) < 40:
        issues.append("log_chunk too short (< 40 chars)")
    if not str(row.get("incident_id", "")).strip():
        issues.append("incident_id missing")
    return len(issues) == 0, issues


def validate_output(analysis: IncidentAnalysis, incident_id: str) -> list[str]:
    warnings = []
    if analysis.incident_severity in ("sev1", "sev2") and analysis.confidence < 50:
        warnings.append(
            f"{incident_id}: severity={analysis.incident_severity} but "
            f"confidence={analysis.confidence} — human review required."
        )
    if not analysis.evidence_lines:
        warnings.append(
            f"{incident_id}: no evidence lines extracted — "
            "log may be too sparse for reliable analysis."
        )
    return warnings


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(row: pd.Series) -> str:
    return f"""You are a senior site reliability engineer and backend developer.

Context:
  Incident ID : {row['incident_id']}
  Service     : {row['service_name']}
  Environment : {row['environment']}
  Region      : {row['region']}

Log chunk:
{row['log_chunk']}

Instructions:
- Base your analysis ONLY on the log content above.
- Do not invent details not present in the logs.
- Evidence lines must quote or closely paraphrase actual log content.
- Immediate actions must be safe (no destructive operations without confirmation).
- incident_severity: sev1=full outage, sev2=major degradation, sev3=minor, sev4=informational.
- If the log is incomplete, say so in what_happened and lower your confidence.
"""


# ── REST call (thin wrapper over shared client) ────────────────────────────

def call_gemini(prompt: str, api_key: str) -> str:
    """Delegates entirely to the shared REST client. No retry logic here."""
    return _rest_call(
        prompt  = prompt,
        api_key = api_key,
        schema  = IncidentAnalysis.model_json_schema(),
        model   = GEMINI_MODEL,
    )


# ── Processing loop ────────────────────────────────────────────────────────

def process_logs(
    df: pd.DataFrame, api_key: str, limit: int, batch_size: int, env_filter: str
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path  = OUTPUT_DIR / "incident_results.jsonl"
    skipped_path  = OUTPUT_DIR / "skipped_incidents.json"
    warnings_path = OUTPUT_DIR / "output_warnings.json"

    if env_filter != "all":
        df = df[df["environment"] == env_filter].reset_index(drop=True)
        print(f"  Filtered to environment='{env_filter}': {len(df):,} rows remaining.")

    results, skipped, all_warnings = [], [], []
    subset = df.head(limit)
    print(f"\nProcessing {len(subset)} incidents…")

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        is_valid, issues = validate_row(row)
        if not is_valid:
            print(f"  [SKIP] {row['incident_id']}: {issues}")
            skipped.append({"incident_id": row["incident_id"], "issues": issues})
            continue

        try:
            raw_json = call_gemini(build_prompt(row), api_key)
        except requests.HTTPError as e:
            print(f"  [HTTP ERROR] {row['incident_id']}: {e}")
            skipped.append({"incident_id": row["incident_id"], "issues": [str(e)]})
            continue
        except requests.Timeout:
            print(f"  [TIMEOUT] {row['incident_id']}")
            skipped.append({"incident_id": row["incident_id"], "issues": ["Timeout"]})
            continue
        except ValueError as e:
            print(f"  [RESPONSE ERROR] {row['incident_id']}: {e}")
            skipped.append({"incident_id": row["incident_id"], "issues": [str(e)]})
            continue

        try:
            analysis = IncidentAnalysis.model_validate_json(raw_json)
        except (ValidationError, ValueError) as e:
            print(f"  [SCHEMA ERROR] {row['incident_id']}: {e}")
            skipped.append({
                "incident_id": row["incident_id"],
                "issues": [f"Schema error: {e}"],
                "raw_response": raw_json[:500],
            })
            continue

        warnings = validate_output(analysis, row["incident_id"])
        if warnings:
            print(f"  [WARN] {warnings}")
            all_warnings.extend(warnings)

        results.append({
            "incident_id": row["incident_id"],
            "service":     row["service_name"],
            **analysis.model_dump(),
        })

        if idx % batch_size == 0 or idx == len(subset):
            print(f"  [{idx}/{len(subset)}] {row['incident_id']} => "
                  f"severity={analysis.incident_severity}  "
                  f"component={analysis.affected_component}  "
                  f"confidence={analysis.confidence}")

    with open(results_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    warnings_path.write_text(json.dumps(all_warnings, indent=2), encoding="utf-8")

    return {
        "processed":     len(results),
        "skipped":       len(skipped),
        "warnings":      len(all_warnings),
        "results_file":  str(results_path),
        "skipped_file":  str(skipped_path),
        "warnings_file": str(warnings_path),
    }


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Log explainer — Gemini REST (requests, no gRPC)")
    parser.add_argument("--limit",      type=int,   default=5,     help="Max incidents to process")
    parser.add_argument("--batch-size", type=int,   default=5,     help="Progress print interval")
    parser.add_argument("--env",        type=str,   default="all",
                        choices=["all","prod","qa","dev"],          help="Filter by environment")
    args = parser.parse_args()

    print("=" * 70)
    print("USE CASE 3: LOG EXPLAINER  (Gemini REST via requests — no gRPC)")
    print("=" * 70)

    print("\n[1/4] Checking GEMINI_API_KEY…")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY first.\n"
              "  Linux/macOS : export GEMINI_API_KEY=your_key\n"
              "  Windows PS  : $env:GEMINI_API_KEY='your_key'")
        sys.exit(1)
    print(f"OK  (length={len(api_key)})")

    print(f"\n[2/4] Loading {CLEAN_DATA_PATH}…")
    if not CLEAN_DATA_PATH.exists():
        print("ERROR: Run  python scripts/prepare_datasets.py  first.")
        sys.exit(1)
    df = pd.read_csv(CLEAN_DATA_PATH)
    require_columns(df, REQUIRED_COLUMNS, "logs_clean")
    require_min_rows(df, 1, "logs_clean")
    print(f"OK  {len(df):,} rows — env filter='{args.env}'.")

    print("\n[3/4] Processing via REST API…")
    print("      NOTE: free-tier = 10 RPM → 6.5 s throttle active.")
    print("      503 auto-fallback: gemini-2.5-flash → 2.0-flash → 1.5-flash")
    summary = process_logs(df, api_key, args.limit, args.batch_size, args.env)

    print("\n[4/4] SUMMARY")
    print("-" * 70)
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")
    print("-" * 70)


if __name__ == "__main__":
    main()
