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
  3. Calls the Gemini REST API directly using the 'requests' library (NOT gRPC).
  4. Post-validates the JSON response against the Pydantic schema.
  5. Saves all results as a JSON Lines (.jsonl) file.
  6. Prints a summary report.

Why requests instead of gRPC?
  The google-genai SDK defaults to gRPC transport, which requires extra
  dependencies (grpcio) and can be tricky to install or run in some
  environments (e.g., restricted networks, Docker-less laptops, or certain
  CI pipelines). Using the Gemini REST API via 'requests' is:
    - Pure HTTP/JSON — works in any Python environment with no native binaries.
    - Easier to debug — you can log and inspect the raw request/response.
    - More portable — same approach works in Java, Node, curl, etc.
    - No gRPC channel management needed.

Gemini REST endpoint used:
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  Authorization: via ?key= query param (API key auth, no OAuth needed)

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
import requests
from pydantic import BaseModel, Field, ValidationError

# ─── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.data_prep import normalize_case, normalize_whitespace, safe_fill
from src.common.validation import require_columns, require_min_rows

# ─── Config ───────────────────────────────────────────────────────────────────
# Gemini REST API base URL.
# We use v1beta because it exposes responseSchema (structured JSON output).
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Flash model is fast and free-tier friendly.
# Replace with "gemini-1.5-pro" for higher accuracy on complex tickets.
GEMINI_MODEL = "gemini-2.0-flash"

# Full REST endpoint for generateContent.
# The API key is appended as a query param — no OAuth needed.
GEMINI_ENDPOINT = f"{GEMINI_REST_BASE}/models/{GEMINI_MODEL}:generateContent"

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "tickets_clean.csv"
OUTPUT_DIR      = PROJECT_ROOT / "output" / "ticket_triage"

REQUIRED_COLUMNS = [
    "ticket_id", "subject", "body", "customer_tier",
    "platform", "region", "app", "priority_hint"
]

# Retry config for transient HTTP 429 / 503 errors.
MAX_RETRIES     = 3
RETRY_BACKOFF_S = 2   # seconds between retries (doubles each attempt)


# ─── Input validation ─────────────────────────────────────────────────────────


def validate_row(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Validate a single ticket row BEFORE sending to Gemini.

    We check here so we never waste an API call on bad data.

    Returns:
        (is_valid: bool, list_of_issues: list[str])
    """
    issues = []

    # Body must have enough content to be meaningful.
    if len(str(row.get("body", "")).strip()) < 20:
        issues.append("body is too short (< 20 chars)")

    # Subject should not be empty.
    if not str(row.get("subject", "")).strip():
        issues.append("subject is blank")

    # ticket_id must be present so we can correlate inputs → outputs.
    if not str(row.get("ticket_id", "")).strip():
        issues.append("ticket_id is missing")

    return len(issues) == 0, issues


# ─── Pydantic output schema ───────────────────────────────────────────────────


class TicketAnalysis(BaseModel):
    """
    Exact shape of the JSON we want Gemini to return.

    We pass TicketAnalysis.model_json_schema() to the Gemini API as
    'responseSchema'. Gemini will then ONLY return JSON that matches
    this shape, making downstream parsing safe and predictable.
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
        description="Confidence score 1–100."
    )


# ─── Output validation ────────────────────────────────────────────────────────


def validate_output(analysis: TicketAnalysis, ticket_id: str) -> list[str]:
    """
    Apply BUSINESS-LEVEL rules on top of Pydantic schema validation.

    Pydantic checks types and literals.
    This function checks whether the values make operational sense.

    Returns a list of warning strings (empty list = all good).
    """
    warnings = []

    # A 'critical' ticket with low confidence should not be auto-routed.
    if analysis.priority == "critical" and analysis.confidence < 60:
        warnings.append(
            f"{ticket_id}: priority=critical but confidence={analysis.confidence} is low. "
            "Review before routing."
        )

    # Draft reply must be non-trivial.
    if len(analysis.suggested_reply.strip()) < 10:
        warnings.append(
            f"{ticket_id}: suggested_reply is too short. Manual reply needed."
        )

    # Bugs without reproduction steps need a follow-up.
    if analysis.category == "bug" and len(analysis.reproduction_steps) == 0:
        warnings.append(
            f"{ticket_id}: category=bug but reproduction_steps is empty. "
            "Ask the reporter for steps."
        )

    return warnings


# ─── Prompt builder ───────────────────────────────────────────────────────────


def build_prompt(row: pd.Series) -> str:
    """
    Build the prompt text from the cleaned ticket row.

    Including structured metadata (tier, app, platform, region) alongside the
    free-text body gives Gemini the full context it needs to route and prioritise
    correctly — without requiring more tokens.
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
- Priority must reflect business impact, not just technical severity.
- The suggested reply must be polite, professional, and actionable.
- Confidence must be an integer from 1 to 100.
"""


# ─── REST API caller ──────────────────────────────────────────────────────────


def build_request_body(prompt: str) -> dict:
    """
    Build the JSON body for the Gemini REST generateContent request.

    Key fields:
      contents            : The conversation turns. We send a single user turn.
      generationConfig
        responseMimeType  : "application/json" tells Gemini to return JSON.
        responseSchema    : Our Pydantic schema converted to JSON Schema.
                            Gemini will constrain its output to this shape.
    """
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            # Tell Gemini to return structured JSON, not prose.
            "responseMimeType": "application/json",
            # Pass our Pydantic model's JSON Schema so Gemini knows
            # the exact field names, types, and allowed enum values.
            "responseSchema": TicketAnalysis.model_json_schema(),
        },
    }


def call_gemini_rest(prompt: str, api_key: str) -> str:
    """
    Call the Gemini REST API using the 'requests' library.

    This function:
      1. Builds the HTTP POST request body.
      2. Sends it to the generateContent endpoint.
      3. Handles HTTP errors and retries on transient failures (429, 503).
      4. Extracts and returns the raw JSON text from the response.

    Returns:
        The JSON string returned by Gemini (to be parsed by Pydantic).

    Raises:
        requests.HTTPError  : For non-retryable HTTP errors.
        ValueError          : If the response structure is unexpected.
    """
    # The API key is passed as a query parameter — no OAuth, no gRPC setup.
    url = f"{GEMINI_ENDPOINT}?key={api_key}"

    # Build the request body according to the Gemini REST spec.
    body = build_request_body(prompt)

    # HTTP headers — we send JSON and expect JSON back.
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Retry loop for transient errors (rate limits, server errors).
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"    [HTTP] POST {GEMINI_ENDPOINT} (attempt {attempt}/{MAX_RETRIES})")

        # Make the HTTP POST request.
        # timeout=(connect_timeout, read_timeout) in seconds.
        response = requests.post(
            url,
            headers=headers,
            json=body,        # requests serialises the dict to JSON automatically
            timeout=(10, 60), # wait 10s to connect, 60s for the model to respond
        )

        # Log the HTTP status code for easy debugging.
        print(f"    [HTTP] Status: {response.status_code}")

        # Handle rate limiting (429) and server errors (503) with backoff.
        if response.status_code in (429, 503) and attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_S * (2 ** (attempt - 1))
            print(f"    [HTTP] Transient error. Waiting {wait}s before retry…")
            time.sleep(wait)
            continue

        # Raise an exception for all other non-2xx responses.
        response.raise_for_status()

        # Parse the response JSON.
        # The Gemini REST response structure is:
        # {
        #   "candidates": [
        #     {
        #       "content": {
        #         "parts": [{"text": "<json string here>"}]
        #       }
        #     }
        #   ]
        # }
        response_json = response.json()

        # Extract the generated text from the nested response structure.
        try:
            generated_text = (
                response_json["candidates"][0]["content"]["parts"][0]["text"]
            )
        except (KeyError, IndexError) as parse_error:
            raise ValueError(
                f"Unexpected response structure from Gemini REST API: {parse_error}. "
                f"Full response: {json.dumps(response_json, indent=2)}"
            )

        print(f"    [HTTP] Received {len(generated_text)} chars of JSON.")
        return generated_text

    raise requests.HTTPError(f"All {MAX_RETRIES} attempts failed.")


# ─── Main processing loop ─────────────────────────────────────────────────────


def process_tickets(df: pd.DataFrame, api_key: str, limit: int, batch_size: int) -> dict:
    """
    Process each cleaned ticket row through the Gemini REST API.

    For each row:
      1. Pre-validate input.
      2. Build the prompt.
      3. Call Gemini via HTTP POST (requests library).
      4. Post-validate the response with Pydantic + business rules.
      5. Collect results.

    Args:
        df         : Clean ticket dataframe.
        api_key    : Gemini API key string.
        limit      : Max number of rows to process.
        batch_size : Print progress every N tickets.

    Returns:
        Summary dict with counts and output file paths.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path  = OUTPUT_DIR / "triage_results.jsonl"
    skipped_path  = OUTPUT_DIR / "skipped_tickets.json"
    warnings_path = OUTPUT_DIR / "output_warnings.json"

    results, skipped, all_warnings = [], [], []

    subset = df.head(limit)
    print(f"\nProcessing {len(subset)} tickets (limit={limit})…")

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):

        # ── Step A: Pre-validate the input row ──────────────────────────────
        is_valid, issues = validate_row(row)
        if not is_valid:
            print(f"  [SKIP] {row['ticket_id']}: {issues}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": issues})
            continue

        # ── Step B: Build the prompt ─────────────────────────────────────────
        prompt = build_prompt(row)

        # ── Step C: Call Gemini REST API via requests ────────────────────────
        try:
            raw_json_text = call_gemini_rest(prompt, api_key)
        except requests.HTTPError as http_err:
            print(f"  [HTTP ERROR] {row['ticket_id']}: {http_err}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": [str(http_err)]})
            continue
        except requests.Timeout:
            print(f"  [TIMEOUT] {row['ticket_id']}: request timed out.")
            skipped.append({"ticket_id": row["ticket_id"], "issues": ["Request timed out"]})
            continue
        except ValueError as val_err:
            print(f"  [RESPONSE ERROR] {row['ticket_id']}: {val_err}")
            skipped.append({"ticket_id": row["ticket_id"], "issues": [str(val_err)]})
            continue

        # ── Step D: Validate the returned JSON with Pydantic ─────────────────
        # Pydantic will raise ValidationError if Gemini returned a field
        # with the wrong type, a disallowed enum value, or a missing field.
        try:
            analysis = TicketAnalysis.model_validate_json(raw_json_text)
        except (ValidationError, ValueError) as schema_err:
            print(f"  [SCHEMA ERROR] {row['ticket_id']}: {schema_err}")
            skipped.append({
                "ticket_id": row["ticket_id"],
                "issues": [f"Schema validation failed: {schema_err}"],
                "raw_response": raw_json_text[:500],  # log snippet for debugging
            })
            continue

        # ── Step E: Apply business-level output rules ────────────────────────
        warnings = validate_output(analysis, row["ticket_id"])
        if warnings:
            print(f"  [WARN] {warnings}")
            all_warnings.extend(warnings)

        # ── Step F: Collect the result ───────────────────────────────────────
        result_dict = {
            "ticket_id":     row["ticket_id"],
            "input_subject": row["subject"],
            **analysis.model_dump(),
        }
        results.append(result_dict)

        # ── Progress update ──────────────────────────────────────────────────
        if idx % batch_size == 0 or idx == len(subset):
            print(f"  [{idx}/{len(subset)}] {row['ticket_id']} => "
                  f"priority={analysis.priority}, "
                  f"team={analysis.likely_team}, "
                  f"confidence={analysis.confidence}")

    # ── Write all outputs ────────────────────────────────────────────────────
    print(f"\nWriting results to {results_path}…")
    with open(results_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    warnings_path.write_text(json.dumps(all_warnings, indent=2), encoding="utf-8")

    return {
        "processed":    len(results),
        "skipped":      len(skipped),
        "warnings":     len(all_warnings),
        "results_file": str(results_path),
        "skipped_file": str(skipped_path),
        "warnings_file": str(warnings_path),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ticket triage using Gemini REST API (no gRPC)"
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Max tickets to process (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Print progress every N tickets (default: 5)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("USE CASE 1: SUPPORT TICKET TRIAGE  (Gemini REST API via requests)")
    print("=" * 70)

    # ── [1/5] Verify API key is set ──────────────────────────────────────────
    print("\n[1/5] Checking GEMINI_API_KEY environment variable…")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        print("Set it with:")
        print("  Linux/macOS : export GEMINI_API_KEY=your_key")
        print("  Windows PS  : $env:GEMINI_API_KEY='your_key'")
        sys.exit(1)
    print(f"OK: API key found (length={len(api_key)}).")

    # ── [2/5] Verify requests is importable ─────────────────────────────────
    print("\n[2/5] Verifying 'requests' library is available…")
    try:
        import requests as req_check
        print(f"OK: requests version {req_check.__version__} is installed.")
    except ImportError:
        print("ERROR: 'requests' is not installed. Run:  pip install requests")
        sys.exit(1)

    # ── [3/5] Load and validate clean data ──────────────────────────────────
    print(f"\n[3/5] Loading clean ticket data from: {CLEAN_DATA_PATH}")
    if not CLEAN_DATA_PATH.exists():
        print("ERROR: Clean data not found.")
        print("Run:  python scripts/prepare_datasets.py")
        sys.exit(1)
    df = pd.read_csv(CLEAN_DATA_PATH)
    require_columns(df, REQUIRED_COLUMNS, "tickets_clean")
    require_min_rows(df, 1, "tickets_clean")
    print(f"OK: Loaded {len(df):,} rows. Will process first {args.limit}.")

    # ── [4/5] Process tickets via REST API ──────────────────────────────────
    print("\n[4/5] Sending tickets to Gemini REST API…")
    summary = process_tickets(df, api_key, args.limit, args.batch_size)

    # ── [5/5] Print summary ──────────────────────────────────────────────────
    print("\n[5/5] SUMMARY")
    print("-" * 70)
    print(f"  Processed   : {summary['processed']}")
    print(f"  Skipped     : {summary['skipped']}")
    print(f"  Warnings    : {summary['warnings']}")
    print(f"  Results     : {summary['results_file']}")
    print(f"  Skipped log : {summary['skipped_file']}")
    print(f"  Warnings    : {summary['warnings_file']}")
    print("-" * 70)
    print("\nDone. Open triage_results.jsonl to review all structured outputs.")


if __name__ == "__main__":
    main()
