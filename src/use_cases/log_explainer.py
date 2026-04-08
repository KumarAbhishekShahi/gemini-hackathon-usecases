"""
log_explainer.py
================

USE CASE 3: Incident Log Explanation using Gemini Structured Output
--------------------------------------------------------------------
Developer pain point:
  Developers and SREs receive raw log dumps and stack traces during an incident.
  They must quickly parse multi-service logs, form a root-cause hypothesis, and
  decide what to fix — often under pressure with incomplete information.

What this script does:
  1. Loads the cleaned logs CSV produced by prepare_datasets.py.
  2. Pre-validates every row before sending to Gemini.
  3. Sends log chunks to Gemini for structured incident analysis.
  4. Post-validates the response against the Pydantic schema.
  5. Saves all results as a JSON Lines (.jsonl) file.
  6. Writes a per-incident runbook in Markdown.

Hackathon demo tip:
  - Show a noisy multi-line log snippet on screen.
  - Run the script live on 1 incident (--limit 1).
  - Show the structured JSON: root cause, evidence, fixes, monitoring checks.
  - Open the generated Markdown runbook — "instantly shareable with the team."

Run:
  python src/use_cases/log_explainer.py
  python src/use_cases/log_explainer.py --limit 3
  python src/use_cases/log_explainer.py --env prod --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# ─── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.data_prep import normalize_whitespace
from src.common.validation import require_columns, require_min_rows

# ─── Config ───────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "logs_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "log_explainer"

REQUIRED_COLUMNS = [
    "incident_id", "log_chunk", "service_name",
    "environment", "incident_title", "region"
]

# ─── Input validation ─────────────────────────────────────────────────────────


def validate_row(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Validate a single log row before sending to Gemini.
    A log chunk must have enough content to reason about.
    Returns (is_valid, list_of_issues).
    """
    issues = []

    log_text = str(row.get("log_chunk", "")).strip()
    if len(log_text) < 40:
        issues.append("log_chunk is too short (< 40 chars) to analyse meaningfully")

    if not str(row.get("incident_id", "")).strip():
        issues.append("incident_id is missing")

    if not str(row.get("service_name", "")).strip():
        issues.append("service_name is missing")

    # Warn if the chunk has no ERROR or WARN level lines.
    if "ERROR" not in log_text.upper() and "WARN" not in log_text.upper():
        issues.append("log_chunk has no ERROR or WARN lines — analysis may be shallow")

    return len(issues) == 0, issues


# ─── Pydantic output schema ───────────────────────────────────────────────────


class IncidentAnalysis(BaseModel):
    """
    Structured incident analysis returned by Gemini.
    Every field is typed and constrained so downstream consumers
    (dashboards, PagerDuty enrichment, Slack alerts) can rely on the shape.
    """

    affected_component: str = Field(
        description="Service, module, or infrastructure component most likely impacted."
    )

    probable_root_cause: str = Field(
        description="Best root-cause hypothesis in one or two sentences."
    )

    confidence: int = Field(
        ge=1, le=100,
        description="Confidence in the root-cause hypothesis, 1-100."
    )

    what_happened: str = Field(
        description="Plain-English summary of the failure sequence."
    )

    evidence_lines: list[str] = Field(
        description="Exact or paraphrased log lines that support the hypothesis."
    )

    immediate_actions: list[str] = Field(
        min_length=1,
        description="Safe first-response actions an on-call engineer should take."
    )

    code_fix_suggestions: list[str] = Field(
        description="Practical engineering fixes for the root cause."
    )

    monitoring_checks: list[str] = Field(
        description="Metrics, alerts, or log queries to add or verify after the fix."
    )

    incident_severity: str = Field(
        pattern=r"^(sev1|sev2|sev3|sev4)$",
        description="Incident severity: sev1 (critical outage) to sev4 (informational)."
    )


# ─── Output validation ────────────────────────────────────────────────────────


def validate_output(analysis: IncidentAnalysis, incident_id: str) -> list[str]:
    """
    Apply business rules on top of Pydantic schema validation.
    These rules catch outputs that are structurally valid but operationally risky.
    Returns a list of warning strings (empty = all clear).
    """
    warnings = []

    # High-severity incidents with low confidence need manual review.
    if analysis.incident_severity in ("sev1", "sev2") and analysis.confidence < 50:
        warnings.append(
            f"{incident_id}: severity={analysis.incident_severity} but confidence="
            f"{analysis.confidence}. Do NOT auto-route — manual review required."
        )

    # Immediate actions must be concrete.
    if len(analysis.immediate_actions) == 0:
        warnings.append(
            f"{incident_id}: no immediate actions returned. "
            "Escalate to on-call lead manually."
        )

    # Evidence lines should reference actual log content.
    if len(analysis.evidence_lines) == 0:
        warnings.append(
            f"{incident_id}: no evidence lines returned. "
            "Root cause may be a hallucination — review carefully."
        )

    # A sev1 incident should always suggest at least one monitoring check.
    if analysis.incident_severity == "sev1" and len(analysis.monitoring_checks) == 0:
        warnings.append(
            f"{incident_id}: sev1 incident with no monitoring_checks. "
            "Add alerting before closing."
        )

    return warnings


# ─── Runbook Markdown writer ──────────────────────────────────────────────────


def write_runbook(analysis: IncidentAnalysis, incident_id: str, out_path: Path) -> None:
    """
    Generate a per-incident runbook in Markdown.
    This can be pasted directly into a post-mortem doc, Confluence, or PagerDuty note.
    """
    lines = [
        f"# Incident Runbook: {incident_id}",
        "",
        f"**Affected component:** {analysis.affected_component}  ",
        f"**Severity:** `{analysis.incident_severity}`  ",
        f"**Confidence:** {analysis.confidence}/100",
        "",
        "---",
        "",
        "## What Happened",
        "",
        analysis.what_happened,
        "",
        "## Probable Root Cause",
        "",
        analysis.probable_root_cause,
        "",
        "## Evidence",
        "",
    ]
    for line in analysis.evidence_lines:
        lines.append(f"- `{line}`")

    lines += [
        "",
        "## Immediate Actions",
        "",
    ]
    for i, action in enumerate(analysis.immediate_actions, start=1):
        lines.append(f"{i}. {action}")

    lines += [
        "",
        "## Code Fix Suggestions",
        "",
    ]
    for fix in analysis.code_fix_suggestions:
        lines.append(f"- {fix}")

    lines += [
        "",
        "## Monitoring Checks to Add",
        "",
    ]
    for check in analysis.monitoring_checks:
        lines.append(f"- {check}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Prompt builder ───────────────────────────────────────────────────────────


def build_prompt(row: pd.Series) -> str:
    """
    Build an incident-analysis prompt that gives Gemini structured context.
    We explicitly instruct the model to be conservative and evidence-based
    so that the output is safe to use in a real incident response workflow.
    """
    return f"""You are a senior site reliability engineer and backend developer.

Context:
  Incident ID : {row['incident_id']}
  Service     : {row['service_name']}
  Environment : {row['environment']}
  Region      : {row['region']}
  Title hint  : {row['incident_title']}

Log chunk:
{row['log_chunk']}

Instructions:
- Base your analysis ONLY on the log content above.
- Do not invent details not present in the logs.
- Evidence lines should quote or closely paraphrase actual log content.
- Immediate actions must be safe (no destructive operations without confirmation).
- Code fix suggestions should be realistic for a backend engineering team.
- Assign incident_severity as: sev1 (full outage), sev2 (major degradation),
  sev3 (minor issue), sev4 (informational).
- If the log is incomplete, say so in what_happened and lower your confidence.
"""


# ─── Main processing loop ─────────────────────────────────────────────────────


def process_logs(df: pd.DataFrame, client, limit: int, env_filter: str | None) -> dict:
    """Process each log row through Gemini and save structured incident analyses."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "incident_analyses.jsonl"
    runbook_dir = OUTPUT_DIR / "runbooks"
    runbook_dir.mkdir(exist_ok=True)
    skipped_path = OUTPUT_DIR / "skipped_incidents.json"
    warnings_path = OUTPUT_DIR / "output_warnings.json"

    if env_filter:
        original_count = len(df)
        df = df[df["environment"] == env_filter].reset_index(drop=True)
        print(f"  Filtered by environment='{env_filter}': {original_count} -> {len(df)} rows")

    results, skipped, all_warnings = [], [], []
    subset = df.head(limit)
    print(f"\nProcessing {len(subset)} log records (limit={limit})...")

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        # ── Pre-validate ─────────────────────────────────────────────────────
        is_valid, issues = validate_row(row)
        if not is_valid:
            print(f"  [SKIP] {row['incident_id']}: {issues}")
            skipped.append({"incident_id": row["incident_id"], "issues": issues})
            continue

        # ── Call Gemini ──────────────────────────────────────────────────────
        prompt = build_prompt(row)
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": IncidentAnalysis.model_json_schema(),
                },
            )
        except Exception as api_error:
            print(f"  [API ERROR] {row['incident_id']}: {api_error}")
            skipped.append({"incident_id": row["incident_id"], "issues": [str(api_error)]})
            continue

        # ── Validate schema ───────────────────────────────────────────────────
        try:
            analysis = IncidentAnalysis.model_validate_json(response.text)
        except (ValidationError, ValueError) as e:
            print(f"  [SCHEMA ERROR] {row['incident_id']}: {e}")
            skipped.append({"incident_id": row["incident_id"], "issues": [str(e)]})
            continue

        # ── Business warnings ─────────────────────────────────────────────────
        warnings = validate_output(analysis, row["incident_id"])
        if warnings:
            print(f"  [WARN] {warnings}")
            all_warnings.extend(warnings)

        # ── Save runbook ──────────────────────────────────────────────────────
        write_runbook(analysis, row["incident_id"],
                      runbook_dir / f"{row['incident_id']}.md")

        # ── Collect result ────────────────────────────────────────────────────
        result_dict = {
            "incident_id": row["incident_id"],
            "service_name": row["service_name"],
            "environment": row["environment"],
            **analysis.model_dump(),
        }
        results.append(result_dict)

        print(f"  [{idx}/{len(subset)}] {row['incident_id']} => "
              f"sev={analysis.incident_severity}, "
              f"component={analysis.affected_component}, "
              f"confidence={analysis.confidence}")

    # ── Write outputs ─────────────────────────────────────────────────────────
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
        "runbook_dir": str(runbook_dir),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Incident log explainer using Gemini")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max log records to process (default: 10)")
    parser.add_argument("--env", type=str, default=None,
                        choices=["dev", "qa", "prod"],
                        help="Filter by environment (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("USE CASE 3: INCIDENT LOG EXPLAINER WITH GEMINI")
    print("=" * 70)

    print("\n[1/5] Checking GEMINI_API_KEY...")
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found. Set it first.")
        sys.exit(1)
    print("OK: API key found.")

    print("\n[2/5] Importing Gemini SDK...")
    try:
        from google import genai
    except ImportError:
        print("ERROR: Run  pip install -U google-genai")
        sys.exit(1)
    client = genai.Client()
    print("OK: Gemini client created.")

    print(f"\n[3/5] Loading clean log data from: {CLEAN_DATA_PATH}")
    if not CLEAN_DATA_PATH.exists():
        print("ERROR: Run  python scripts/prepare_datasets.py  first.")
        sys.exit(1)
    df = pd.read_csv(CLEAN_DATA_PATH)
    require_columns(df, REQUIRED_COLUMNS, "logs_clean")
    require_min_rows(df, 1, "logs_clean")
    print(f"OK: Loaded {len(df)} rows. Processing first {args.limit}.")

    print("\n[4/5] Sending log chunks to Gemini...")
    summary = process_logs(df, client, args.limit, args.env)

    print("\n[5/5] SUMMARY")
    print("-" * 70)
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print("-" * 70)
    print("\nDone. Check incident_analyses.jsonl and the runbooks/ folder.")


if __name__ == "__main__":
    main()
