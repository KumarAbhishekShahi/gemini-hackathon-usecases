"""
output_validation.py
====================

Standalone validation script that audits ALL generated output files
from the three Gemini use cases in one pass.

Run this after all three application scripts have completed to get a
consolidated quality report before submitting or presenting.

Run:
  python scripts/output_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT  = PROJECT_ROOT / "output"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load a .jsonl file into a list of dicts. Returns [] if file is missing."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [JSONL PARSE ERROR] {path.name}: {e}")
    return records

def load_json(path: Path) -> Any:
    """Load a regular JSON file. Returns None if missing."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)

def check(label: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)

# ─── Use Case 1: Ticket Triage ────────────────────────────────────────────────

def validate_ticket_triage() -> int:
    section("USE CASE 1: Ticket Triage Outputs")
    results_path  = OUTPUT_ROOT / "ticket_triage" / "triage_results.jsonl"
    skipped_path  = OUTPUT_ROOT / "ticket_triage" / "skipped_tickets.json"
    warnings_path = OUTPUT_ROOT / "ticket_triage" / "output_warnings.json"

    failures = 0

    records  = load_jsonl(results_path)
    skipped  = load_json(skipped_path)  or []
    warnings = load_json(warnings_path) or []

    check("results file exists",   results_path.exists())
    check("skipped log exists",    skipped_path.exists())
    check("warnings log exists",   warnings_path.exists())

    if not records:
        print("  [INFO] No results yet. Run ticket_triage.py first.")
        return 0

    check(f"at least 1 record returned", len(records) >= 1,
          f"{len(records)} records")

    # Required fields in every record
    required_fields = [
        "ticket_id", "category", "priority", "likely_team",
        "short_summary", "suggested_reply", "confidence"
    ]
    missing_fields_count = 0
    invalid_confidence   = 0
    empty_reply          = 0

    valid_categories = {"bug","feature_request","access_issue","billing","how_to","other"}
    valid_priorities  = {"low","medium","high","critical"}
    valid_teams       = {"frontend","backend","qa","devops","security","support","product","unknown"}

    invalid_category = 0
    invalid_priority = 0
    invalid_team     = 0

    for record in records:
        for field in required_fields:
            if field not in record:
                missing_fields_count += 1

        conf = record.get("confidence", 0)
        if not (1 <= int(conf) <= 100):
            invalid_confidence += 1

        reply = str(record.get("suggested_reply", "")).strip()
        if len(reply) < 10:
            empty_reply += 1

        if record.get("category") not in valid_categories:
            invalid_category += 1

        if record.get("priority") not in valid_priorities:
            invalid_priority += 1

        if record.get("likely_team") not in valid_teams:
            invalid_team += 1

    check("no missing required fields",  missing_fields_count == 0,
          f"{missing_fields_count} missing")
    check("all confidence scores 1-100", invalid_confidence == 0,
          f"{invalid_confidence} out of range")
    check("all replies non-empty",       empty_reply == 0,
          f"{empty_reply} empty")
    check("all categories valid",        invalid_category == 0,
          f"{invalid_category} invalid")
    check("all priorities valid",        invalid_priority == 0,
          f"{invalid_priority} invalid")
    check("all teams valid",             invalid_team == 0,
          f"{invalid_team} invalid")

    print(f"\n  Skipped: {len(skipped)}   Warnings: {len(warnings)}")
    failures += sum([missing_fields_count, invalid_confidence, empty_reply,
                     invalid_category, invalid_priority, invalid_team])
    return failures

# ─── Use Case 2: Sprint Planner ───────────────────────────────────────────────

def validate_sprint_planner() -> int:
    section("USE CASE 2: Sprint Planner Outputs")
    results_path  = OUTPUT_ROOT / "sprint_planner" / "sprint_plans.jsonl"
    md_dir        = OUTPUT_ROOT / "sprint_planner" / "markdown"
    warnings_path = OUTPUT_ROOT / "sprint_planner" / "output_warnings.json"

    failures = 0

    records  = load_jsonl(results_path)
    warnings = load_json(warnings_path) or []

    check("results file exists",  results_path.exists())
    check("markdown dir exists",  md_dir.exists() if md_dir.exists() else False)

    if not records:
        print("  [INFO] No results yet. Run sprint_planner.py first.")
        return 0

    check(f"at least 1 record returned", len(records) >= 1, f"{len(records)} records")

    no_stories        = 0
    no_ac             = 0
    zero_points       = 0
    missing_epic      = 0
    no_goal           = 0
    md_files          = list(md_dir.glob("*.md")) if md_dir.exists() else []

    for record in records:
        if not record.get("epic_name", "").strip():
            missing_epic += 1
        if not record.get("business_goal", "").strip():
            no_goal += 1

        stories = record.get("stories", [])
        if len(stories) == 0:
            no_stories += 1
        for story in stories:
            if len(story.get("acceptance_criteria", [])) == 0:
                no_ac += 1
            if int(story.get("estimate_points", 0)) == 0:
                zero_points += 1

    check("all records have stories",          no_stories == 0,
          f"{no_stories} empty")
    check("all stories have acceptance criteria", no_ac == 0,
          f"{no_ac} missing")
    check("all stories have story points > 0", zero_points == 0,
          f"{zero_points} zero")
    check("all records have epic name",        missing_epic == 0,
          f"{missing_epic} missing")
    check("markdown files generated",          len(md_files) > 0,
          f"{len(md_files)} files in markdown/")

    print(f"\n  Warnings: {len(warnings)}")
    failures += sum([no_stories, no_ac, zero_points, missing_epic])
    return failures

# ─── Use Case 3: Log Explainer ────────────────────────────────────────────────

def validate_log_explainer() -> int:
    section("USE CASE 3: Log Explainer Outputs")
    results_path  = OUTPUT_ROOT / "log_explainer" / "incident_analyses.jsonl"
    runbook_dir   = OUTPUT_ROOT / "log_explainer" / "runbooks"
    warnings_path = OUTPUT_ROOT / "log_explainer" / "output_warnings.json"

    failures = 0

    records  = load_jsonl(results_path)
    warnings = load_json(warnings_path) or []

    check("results file exists",  results_path.exists())
    check("runbooks dir exists",  runbook_dir.exists() if runbook_dir.exists() else False)

    if not records:
        print("  [INFO] No results yet. Run log_explainer.py first.")
        return 0

    check(f"at least 1 record returned", len(records) >= 1, f"{len(records)} records")

    valid_severities     = {"sev1","sev2","sev3","sev4"}
    invalid_severity     = 0
    empty_root_cause     = 0
    no_immediate_actions = 0
    no_evidence          = 0
    bad_confidence       = 0

    runbook_files = list(runbook_dir.glob("*.md")) if runbook_dir.exists() else []

    for record in records:
        if record.get("incident_severity") not in valid_severities:
            invalid_severity += 1
        if not str(record.get("probable_root_cause", "")).strip():
            empty_root_cause += 1
        if len(record.get("immediate_actions", [])) == 0:
            no_immediate_actions += 1
        if len(record.get("evidence_lines", [])) == 0:
            no_evidence += 1
        conf = record.get("confidence", 0)
        if not (1 <= int(conf) <= 100):
            bad_confidence += 1

    check("all severity values valid",      invalid_severity == 0,
          f"{invalid_severity} invalid")
    check("all records have root cause",    empty_root_cause == 0,
          f"{empty_root_cause} empty")
    check("all records have actions",       no_immediate_actions == 0,
          f"{no_immediate_actions} missing")
    check("all records have evidence",      no_evidence == 0,
          f"{no_evidence} missing")
    check("all confidence scores 1-100",    bad_confidence == 0,
          f"{bad_confidence} out of range")
    check("runbook files generated",        len(runbook_files) > 0,
          f"{len(runbook_files)} files in runbooks/")

    print(f"\n  Warnings: {len(warnings)}")
    failures += sum([invalid_severity, empty_root_cause, no_immediate_actions,
                     no_evidence, bad_confidence])
    return failures

# ─── Summary ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  OUTPUT VALIDATION — GEMINI HACKATHON USE CASES")
    print("=" * 70)

    total_failures = 0
    total_failures += validate_ticket_triage()
    total_failures += validate_sprint_planner()
    total_failures += validate_log_explainer()

    section("FINAL RESULT")
    if total_failures == 0:
        print("  ALL CHECKS PASSED. Results are ready for demo or submission.")
    else:
        print(f"  {total_failures} CHECK(S) FAILED.")
        print("  Review the FAIL lines above and re-run the relevant use case.")
    print()

if __name__ == "__main__":
    main()
