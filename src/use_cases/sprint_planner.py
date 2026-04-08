"""
sprint_planner.py
=================
USE CASE 2: Requirement to Sprint Plan using Gemini Structured Output.

Calls Gemini via the shared REST client (src/common/gemini_client.py).
No gRPC. No google-genai SDK. Pure HTTP via 'requests'.

Run:
  python src/use_cases/sprint_planner.py
  python src/use_cases/sprint_planner.py --limit 5
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
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "requirements_clean.csv"
OUTPUT_DIR      = PROJECT_ROOT / "output" / "sprint_planner"
REQUIRED_COLUMNS = [
    "requirement_id", "raw_requirement_text", "product_area",
    "requester_role", "business_context", "delivery_window",
]


# ── Pydantic schema ────────────────────────────────────────────────────────

class Story(BaseModel):
    title: str
    description: str
    acceptance_criteria: list[str]
    test_cases: list[str]
    dependencies: list[str]
    estimate_points: int = Field(ge=1, le=21)

class SprintPlan(BaseModel):
    epic_name: str
    business_goal: str
    stories: list[Story] = Field(min_length=1)
    demo_plan: list[str]
    risk_flags: list[str]


# ── Input / output validation ──────────────────────────────────────────────

def validate_row(row: pd.Series) -> tuple[bool, list[str]]:
    issues = []
    if len(str(row.get("raw_requirement_text", "")).strip()) < 30:
        issues.append("requirement text too short (< 30 chars)")
    if not str(row.get("requirement_id", "")).strip():
        issues.append("requirement_id missing")
    return len(issues) == 0, issues


def validate_output(plan: SprintPlan, req_id: str) -> list[str]:
    warnings = []
    total_pts = sum(s.estimate_points for s in plan.stories)
    if total_pts > 80:
        warnings.append(f"{req_id}: total points={total_pts} — may be too large for one sprint.")
    if len(plan.stories) > 10:
        warnings.append(f"{req_id}: {len(plan.stories)} stories — consider splitting into multiple sprints.")
    return warnings


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(row: pd.Series) -> str:
    return f"""You are a senior product-minded engineering lead.

Context:
  Requirement ID  : {row['requirement_id']}
  Product area    : {row['product_area']}
  Requested by    : {row['requester_role']}
  Business context: {row['business_context']}
  Delivery window : {row['delivery_window']}

Raw requirement:
{row['raw_requirement_text']}

Instructions:
- Break into realistic sprint stories with acceptance criteria and test cases.
- Acceptance criteria must be testable statements.
- Use Fibonacci story points (1,2,3,5,8,13,21).
- Flag unclear items in risk_flags rather than inventing details.
- Output only the requested structured data.
"""


# ── REST call (thin wrapper over shared client) ────────────────────────────

def call_gemini(prompt: str, api_key: str) -> str:
    """Delegates entirely to the shared REST client. No retry logic here."""
    return _rest_call(
        prompt  = prompt,
        api_key = api_key,
        schema  = SprintPlan.model_json_schema(),
        model   = GEMINI_MODEL,
    )


# ── Processing loop ────────────────────────────────────────────────────────

def process_requirements(df: pd.DataFrame, api_key: str, limit: int, batch_size: int) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path  = OUTPUT_DIR / "sprint_plan_results.jsonl"
    skipped_path  = OUTPUT_DIR / "skipped_requirements.json"
    warnings_path = OUTPUT_DIR / "output_warnings.json"

    results, skipped, all_warnings = [], [], []
    subset = df.head(limit)
    print(f"\nProcessing {len(subset)} requirements…")

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        is_valid, issues = validate_row(row)
        if not is_valid:
            print(f"  [SKIP] {row['requirement_id']}: {issues}")
            skipped.append({"requirement_id": row["requirement_id"], "issues": issues})
            continue

        try:
            raw_json = call_gemini(build_prompt(row), api_key)
        except requests.HTTPError as e:
            print(f"  [HTTP ERROR] {row['requirement_id']}: {e}")
            skipped.append({"requirement_id": row["requirement_id"], "issues": [str(e)]})
            continue
        except requests.Timeout:
            print(f"  [TIMEOUT] {row['requirement_id']}")
            skipped.append({"requirement_id": row["requirement_id"], "issues": ["Timeout"]})
            continue
        except ValueError as e:
            print(f"  [RESPONSE ERROR] {row['requirement_id']}: {e}")
            skipped.append({"requirement_id": row["requirement_id"], "issues": [str(e)]})
            continue

        try:
            plan = SprintPlan.model_validate_json(raw_json)
        except (ValidationError, ValueError) as e:
            print(f"  [SCHEMA ERROR] {row['requirement_id']}: {e}")
            skipped.append({
                "requirement_id": row["requirement_id"],
                "issues": [f"Schema error: {e}"],
                "raw_response": raw_json[:500],
            })
            continue

        warnings = validate_output(plan, row["requirement_id"])
        if warnings:
            print(f"  [WARN] {warnings}")
            all_warnings.extend(warnings)

        total_pts = sum(s.estimate_points for s in plan.stories)
        results.append({
            "requirement_id": row["requirement_id"],
            **plan.model_dump(),
        })

        if idx % batch_size == 0 or idx == len(subset):
            print(f"  [{idx}/{len(subset)}] {row['requirement_id']} => "
                  f"epic='{plan.epic_name}'  stories={len(plan.stories)}  pts={total_pts}")

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
    parser = argparse.ArgumentParser(description="Sprint planner — Gemini REST (requests, no gRPC)")
    parser.add_argument("--limit",      type=int, default=5,  help="Max requirements to process")
    parser.add_argument("--batch-size", type=int, default=5,  help="Progress print interval")
    args = parser.parse_args()

    print("=" * 70)
    print("USE CASE 2: SPRINT PLANNER  (Gemini REST via requests — no gRPC)")
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
    require_columns(df, REQUIRED_COLUMNS, "requirements_clean")
    require_min_rows(df, 1, "requirements_clean")
    print(f"OK  {len(df):,} rows — will process first {args.limit}.")

    print("\n[3/4] Processing via REST API…")
    print("      NOTE: free-tier = 10 RPM → 6.5 s throttle active.")
    print("      503 auto-fallback: gemini-2.5-flash → 2.0-flash → 1.5-flash")
    summary = process_requirements(df, api_key, args.limit, args.batch_size)

    print("\n[4/4] SUMMARY")
    print("-" * 70)
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")
    print("-" * 70)


if __name__ == "__main__":
    main()
