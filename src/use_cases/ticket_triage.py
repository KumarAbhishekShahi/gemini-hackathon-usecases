"""
ticket_triage.py
================

USE CASE 1: Support Ticket Triage using Gemini Structured Output
-----------------------------------------------------------------
Calls Gemini via the REST API using 'requests' (no gRPC).
429 handling is in src/common/gemini_client.py — see there for:
  - retryDelay parsing from response body
  - exponential backoff with jitter
  - inter-request throttle (6.5 s gap for 10 RPM free tier)

Run:
  python src/use_cases/ticket_triage.py
  python src/use_cases/ticket_triage.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

# ─── Path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ─── Shared REST client with 429 / retry / throttle logic ─────────────────
# All backoff, jitter, retryDelay parsing, and rate-limit throttling lives
# in gemini_client.py — one fix applies to every use-case script.
from src.common.gemini_client import call_gemini_rest as _rest_call
from src.common.data_prep import normalize_case, normalize_whitespace, safe_fill
from src.common.validation import require_columns, require_min_rows

# ─── Config ────────────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-2.0-flash"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "tickets_clean.csv"
OUTPUT_DIR      = PROJECT_ROOT / "output" / "ticket_triage"
REQUIRED_COLUMNS = [
    "ticket_id", "subject", "body", "customer_tier",
    "platform", "region", "app", "priority_hint",
]


# ─── Pydantic output schema ────────────────────────────────────────────────

class TicketAnalysis(BaseModel):
    category: Literal[
        "bug", "feature_request", "access_issue", "billing", "how_to", "other"
    ] = Field(description="Best matching category.")

    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency based on business impact."
    )

    likely_team: Literal[
        "frontend", "backend", "qa", "devops", "security",
        "support", "product", "unknown"
    ] = Field(description="Team most likely to investigate first.")

    short_summary: str = Field(description="One-line summary.")
    customer_visible_impact: str = Field(description="User impact in 1-2 sentences.")
    reproduction_steps: list[str] = Field(description="Steps to reproduce, or empty list.")
    suggested_reply: str = Field(description="Polite, actionable reply for support.")
    confidence: int = Field(ge=1, le=100, description="Confidence score 1-100.")


# ─── Input / output validation ────────────────────────────────────────────

def validate_row(row: pd.Series) -> tuple[bool, list[str]]:
    issues = []
    if len(str(row.get("body", "")).strip()) < 20:
        issues.append("body too short (< 20 chars)")
    if not str(row.get("subject", "")).strip():
        issues.append("subject is blank")
    if not str(row.get("ticket_id", "")).strip():
        issues.append("ticket_id missing")
    return len(issues) == 0, issues


def validate_output(analysis: TicketAnalysis, ticket_id: str) -> list[str]:
    warnings = []
    if analysis.priority == "critical" and analysis.confidence < 60:
        warnings.append(f"{ticket_id}: critical priority but low confidence — review before routing.")
    if len(analysis.suggested_reply.strip()) < 10:
        warnings.append(f"{ticket_id}: suggested_reply too short — needs manual reply.")
    if analysis.category == "bug" and not analysis.reproduction_steps:
        warnings.append(f"{ticket_id}: bug without reproduction steps — ask reporter.")
    return warnings


# ─── Prompt builder ───────────────────────────────────────────────────────

def build_prompt(row: pd.Series) -> str:
    return f"""You are a senior support triage assistant for a SaaS software company.

Context:
  Ticket ID  : {row['ticket_id']}
  Customer   : {row['customer_tier']} tier
  App module : {row['app']}
  Platform   : {row['platform']}
  Region     : {row['region']}
  Hint       : {row['priority_hint']}

Subject:
{row['subject']}

Body:
{row['body']}

Instructions:
- Classify the ticket using only information provided. Be conservative.
- Do not invent reproduction steps that are not stated.
- Priority must reflect business impact.
- suggested_reply must be polite, professional, and actionable.
- confidence is an integer 1-100.
"""


# ─── REST API call (thin wrapper over shared client) ──────────────────────

def call_gemini(prompt: str, api_key: str) -> str:
    """
    Call Gemini via the shared REST client.
    Retry / 429 / throttle logic is all in src/common/gemini_client.py.
    """
    return _rest_call(
        prompt  = prompt,
        api_key = api_key,
        schema  = TicketAnalysis.model_json_schema(),
        model   = GEMINI_MODEL,
    )


# ─── Processing loop ──────────────────────────────────────────────────────

def process_tickets(df: pd.DataFrame, api_key: str, limit: int, batch_size: int) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path  = OUTPUT_DIR / "triage_results.jsonl"
    skipped_path  = OUTPUT_DIR / "skipped_tickets.json"
    warnings_path = OUTPUT_DIR / "output_warnings.json"

    results, skipped, all_warnings = [], [], []
    subset = df.head(limit)
    print(f"\nProcessing {len(subset)} tickets …")

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        # Pre-validate input
        is_valid, issues = validate_row(row)
        if not is_valid:
            print(f"  [SKIP] {row['ticket_id']}: {issues}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": issues})
            continue

        # Call Gemini (shared client handles all 429 logic)
        try:
            raw_json = call_gemini(build_prompt(row), api_key)
        except requests.HTTPError as e:
            print(f"  [HTTP ERROR] {row['ticket_id']}: {e}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": [str(e)]})
            continue
        except requests.Timeout:
            print(f"  [TIMEOUT] {row['ticket_id']}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": ["Timeout"]})
            continue
        except ValueError as e:
            print(f"  [RESPONSE ERROR] {row['ticket_id']}: {e}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": [str(e)]})
            continue

        # Validate with Pydantic
        try:
            analysis = TicketAnalysis.model_validate_json(raw_json)
        except (ValidationError, ValueError) as e:
            print(f"  [SCHEMA ERROR] {row['ticket_id']}: {e}")
            skipped.append({
                "ticket_id": row["ticket_id"],
                "issues": [f"Schema error: {e}"],
                "raw_response": raw_json[:500],
            })
            continue

        # Business-rule warnings
        warnings = validate_output(analysis, row["ticket_id"])
        if warnings:
            print(f"  [WARN] {warnings}")
            all_warnings.extend(warnings)

        results.append({"ticket_id": row["ticket_id"], "input_subject": row["subject"], **analysis.model_dump()})

        if idx % batch_size == 0 or idx == len(subset):
            print(f"  [{idx}/{len(subset)}] {row['ticket_id']} => "
                  f"priority={analysis.priority}  team={analysis.likely_team}  confidence={analysis.confidence}")

    # Write outputs
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


# ─── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket triage — Gemini REST (requests, no gRPC)")
    parser.add_argument("--limit",      type=int, default=10, help="Max tickets to process")
    parser.add_argument("--batch-size", type=int, default=5,  help="Progress print interval")
    args = parser.parse_args()

    print("=" * 70)
    print("USE CASE 1: TICKET TRIAGE  (Gemini REST via requests — no gRPC)")
    print("=" * 70)

    print("\n[1/5] Checking GEMINI_API_KEY …")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.\n"
              "  Linux/macOS : export GEMINI_API_KEY=your_key\n"
              "  Windows PS  : $env:GEMINI_API_KEY='your_key'")
        sys.exit(1)
    print(f"OK  (length={len(api_key)})")

    print("\n[2/5] Checking 'requests' library …")
    import requests as _r
    print(f"OK  requests=={_r.__version__}")

    print(f"\n[3/5] Loading {CLEAN_DATA_PATH} …")
    if not CLEAN_DATA_PATH.exists():
        print("ERROR: Run  python scripts/prepare_datasets.py  first.")
        sys.exit(1)
    df = pd.read_csv(CLEAN_DATA_PATH)
    require_columns(df, REQUIRED_COLUMNS, "tickets_clean")
    require_min_rows(df, 1, "tickets_clean")
    print(f"OK  {len(df):,} rows — will process first {args.limit}.")

    print("\n[4/5] Processing tickets …")
    print("      NOTE: free-tier = 10 RPM → 6.5 s inter-request throttle active.")
    print("      If you get 429 errors the shared client will wait up to 120 s per retry.")
    summary = process_tickets(df, api_key, args.limit, args.batch_size)

    print("\n[5/5] SUMMARY")
    print("-" * 70)
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")
    print("-" * 70)


if __name__ == "__main__":
    main()
