"""
ticket_triage.py
================

USE CASE 1: Support Ticket Triage using Gemini Structured Output
-----------------------------------------------------------------
Developer pain point:
  Support inboxes, issue trackers, and Slack channels receive raw unstructured
  ticket text. A developer or support lead must manually read each one, decide
  priority, pick the right team, and draft a reply.

What this script does:
  1. Loads the cleaned ticket CSV produced by prepare_datasets.py.
  2. Pre-validates every row before sending it to Gemini.
  3. Sends each ticket to Gemini with a structured schema.
  4. Post-validates the JSON response against the Pydantic schema.
  5. Saves all results as a JSON Lines (.jsonl) file.
  6. Prints a summary report.

Hackathon demo tip:
  - Show 3 raw ticket rows (messy, unformatted text).
  - Run the script live on just those 3 rows (use --limit 3).
  - Reveal the structured JSON output with priority, team, and draft reply.

Run:
  python src/use_cases/ticket_triage.py
  python src/use_cases/ticket_triage.py --limit 5
  python src/use_cases/ticket_triage.py --batch-size 10 --limit 50
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
from pydantic import BaseModel, Field, ValidationError

# ─── Path setup ──────────────────────────────────────────────────────────────
# We need to make sure Python can find the shared utilities in src/common.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.data_prep import normalize_case, normalize_whitespace, safe_fill
from src.common.validation import require_columns, require_min_rows

# ─── Config ───────────────────────────────────────────────────────────────────
# Switch to "gemini-2.0-flash" if this preview model is unavailable.
GEMINI_MODEL = "gemini-2.0-flash"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "tickets_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ticket_triage"

REQUIRED_COLUMNS = ["ticket_id", "subject", "body", "customer_tier",
                    "platform", "region", "app", "priority_hint"]

# ─── Input validation ─────────────────────────────────────────────────────────


def validate_row(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Validate a single ticket row before sending to Gemini.

    Returns:
        A tuple of (is_valid: bool, list_of_issues: list[str]).
    """
    issues = []

    # The body must have enough content to be useful.
    if len(str(row.get("body", "")).strip()) < 20:
        issues.append("body is too short (< 20 chars)")

    # Subject should not be empty.
    if not str(row.get("subject", "")).strip():
        issues.append("subject is blank")

    # ticket_id must be present.
    if not str(row.get("ticket_id", "")).strip():
        issues.append("ticket_id is missing")

    return len(issues) == 0, issues


# ─── Pydantic output schema ───────────────────────────────────────────────────


class TicketAnalysis(BaseModel):
    """
    Exact shape of the JSON we want Gemini to return.
    Pydantic validates every field type and literal value.
    """

    category: Literal[
        "bug", "feature_request", "access_issue", "billing", "how_to", "other"
    ] = Field(description="Best matching category for the ticket.")

    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency level based on impact and severity."
    )

    likely_team: Literal[
        "frontend", "backend", "qa", "devops", "security",
        "support", "product", "unknown"
    ] = Field(description="Team most likely to investigate first.")

    short_summary: str = Field(
        description="One-line summary in plain business language."
    )

    customer_visible_impact: str = Field(
        description="User impact in one or two sentences."
    )

    reproduction_steps: list[str] = Field(
        description="Steps to reproduce. Empty list if unknown."
    )

    suggested_reply: str = Field(
        description="A polite, concise reply that support can send back."
    )

    confidence: int = Field(
        ge=1, le=100,
        description="Confidence score 1-100."
    )


# ─── Output validation ────────────────────────────────────────────────────────


def validate_output(analysis: TicketAnalysis, ticket_id: str) -> list[str]:
    """
    Apply business-level rules on top of Pydantic schema validation.

    These rules catch logically inconsistent outputs that are technically valid
    JSON but would cause problems downstream.

    Returns a list of warning strings (empty list = all good).
    """
    warnings = []

    # A 'critical' priority ticket must have a high confidence score.
    if analysis.priority == "critical" and analysis.confidence < 60:
        warnings.append(
            f"{ticket_id}: priority=critical but confidence={analysis.confidence} is low. "
            "Review before routing."
        )

    # Suggested reply should not be empty.
    if len(analysis.suggested_reply.strip()) < 10:
        warnings.append(f"{ticket_id}: suggested_reply is too short. Manual review needed.")

    # If category is 'bug', reproduction_steps should ideally not be empty.
    if analysis.category == "bug" and len(analysis.reproduction_steps) == 0:
        warnings.append(
            f"{ticket_id}: category=bug but reproduction_steps is empty. "
            "Ask the reporter for steps."
        )

    return warnings


# ─── Prompt builder ──────────────────────────────────────────────────────────


def build_prompt(row: pd.Series) -> str:
    """
    Build a focused prompt using the cleaned ticket fields.
    Including structured fields (tier, app, platform) helps Gemini make better
    routing and priority decisions without requiring more input tokens.
    """
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
- Classify the ticket accurately using only the information provided.
- Be conservative: do not invent details or reproduction steps.
- Priority should reflect business impact, not just technical severity.
- The suggested reply should be polite, professional, and actionable.
- Confidence must be an integer from 1 to 100.
"""


# ─── Main processing loop ─────────────────────────────────────────────────────


def process_tickets(df: pd.DataFrame, client, limit: int, batch_size: int) -> dict:
    """
    Process each ticket row through Gemini and collect structured results.

    Args:
        df: Clean ticket dataframe.
        client: Initialized Gemini client.
        limit: Maximum number of tickets to process.
        batch_size: Print a progress update every N tickets.

    Returns:
        A summary dictionary with counts and file paths.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "triage_results.jsonl"
    skipped_path = OUTPUT_DIR / "skipped_tickets.json"
    warnings_path = OUTPUT_DIR / "output_warnings.json"

    results = []
    skipped = []
    all_warnings = []

    subset = df.head(limit)
    print(f"\nProcessing {len(subset)} tickets (limit={limit})...")

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        # ── Pre-validate row ─────────────────────────────────────────────────
        is_valid, issues = validate_row(row)
        if not is_valid:
            print(f"  [SKIP] {row['ticket_id']}: {issues}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": issues})
            continue

        # ── Build prompt ─────────────────────────────────────────────────────
        prompt = build_prompt(row)

        # ── Call Gemini ──────────────────────────────────────────────────────
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": TicketAnalysis.model_json_schema(),
                },
            )
        except Exception as api_error:
            print(f"  [API ERROR] {row['ticket_id']}: {api_error}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": [str(api_error)]})
            continue

        # ── Validate output with Pydantic ────────────────────────────────────
        try:
            analysis = TicketAnalysis.model_validate_json(response.text)
        except (ValidationError, ValueError) as validation_error:
            print(f"  [SCHEMA ERROR] {row['ticket_id']}: {validation_error}")
            skipped.append({
                "ticket_id": row["ticket_id"],
                "issues": [f"Schema validation failed: {validation_error}"]
            })
            continue

        # ── Business-level output warnings ───────────────────────────────────
        warnings = validate_output(analysis, row["ticket_id"])
        if warnings:
            print(f"  [WARN] {warnings}")
            all_warnings.extend(warnings)

        # ── Collect result ───────────────────────────────────────────────────
        result_dict = {
            "ticket_id": row["ticket_id"],
            "input_subject": row["subject"],
            **analysis.model_dump(),
        }
        results.append(result_dict)

        # ── Progress logging ─────────────────────────────────────────────────
        if idx % batch_size == 0 or idx == len(subset):
            print(f"  [{idx}/{len(subset)}] latest: {row['ticket_id']} "
                  f"=> priority={analysis.priority}, team={analysis.likely_team}, "
                  f"confidence={analysis.confidence}")

    # ── Write outputs ────────────────────────────────────────────────────────
    with open(results_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    warnings_path.write_text(json.dumps(all_warnings, indent=2), encoding="utf-8")

    return {
        "processed": len(results),
        "skipped": len(skipped),
        "warnings": len(all_warnings),
        "results_file": str(results_path),
        "skipped_file": str(skipped_path),
        "warnings_file": str(warnings_path),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket triage using Gemini")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max tickets to process (default: 10)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Print progress every N tickets (default: 5)")
    args = parser.parse_args()

    print("=" * 70)
    print("USE CASE 1: SUPPORT TICKET TRIAGE WITH GEMINI")
    print("=" * 70)

    # ── Check API key ────────────────────────────────────────────────────────
    print("\n[1/5] Checking GEMINI_API_KEY...")
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in environment.")
        print("Set it with:  export GEMINI_API_KEY=your_key")
        sys.exit(1)
    print("OK: API key found.")

    # ── Import Gemini SDK ────────────────────────────────────────────────────
    print("\n[2/5] Importing Gemini SDK...")
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai is not installed.")
        print("Run:  pip install -U google-genai")
        sys.exit(1)
    client = genai.Client()
    print("OK: Gemini client created.")

    # ── Load and validate clean data ─────────────────────────────────────────
    print(f"\n[3/5] Loading clean ticket data from: {CLEAN_DATA_PATH}")
    if not CLEAN_DATA_PATH.exists():
        print("ERROR: Clean data not found. Run  python scripts/prepare_datasets.py  first.")
        sys.exit(1)
    df = pd.read_csv(CLEAN_DATA_PATH)
    require_columns(df, REQUIRED_COLUMNS, "tickets_clean")
    require_min_rows(df, 1, "tickets_clean")
    print(f"OK: Loaded {len(df)} rows. Processing first {args.limit}.")

    # ── Process ──────────────────────────────────────────────────────────────
    print("\n[4/5] Sending tickets to Gemini...")
    summary = process_tickets(df, client, args.limit, args.batch_size)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n[5/5] SUMMARY")
    print("-" * 70)
    print(f"  Processed  : {summary['processed']}")
    print(f"  Skipped    : {summary['skipped']}")
    print(f"  Warnings   : {summary['warnings']}")
    print(f"  Results    : {summary['results_file']}")
    print(f"  Skipped log: {summary['skipped_file']}")
    print(f"  Warnings   : {summary['warnings_file']}")
    print("-" * 70)
    print("\nDone. Open triage_results.jsonl to see all structured outputs.")


if __name__ == "__main__":
    main()
