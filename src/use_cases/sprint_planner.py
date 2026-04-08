"""
sprint_planner.py
=================

USE CASE 2: Requirement-to-Sprint Plan using Gemini Structured Output
----------------------------------------------------------------------
Developer pain point:
  PMs, clients, and ops leads send vague requirement text via email, Notion,
  Slack, or meeting notes. A developer lead must manually convert these into
  user stories, acceptance criteria, test cases, and story point estimates.

What this script does:
  1. Loads the cleaned requirements CSV produced by prepare_datasets.py.
  2. Pre-validates every row before sending to Gemini.
  3. Builds a structured sprint plan (epic, stories, criteria, tests, points).
  4. Post-validates the response against the Pydantic schema.
  5. Saves all results as a JSON Lines (.jsonl) file.
  6. Writes a Markdown summary that can be copied into a wiki or Confluence.

Hackathon demo tip:
  - Paste one raw requirement text on screen.
  - Run the script live on 1 record (--limit 1).
  - Show the full sprint plan with stories, criteria, and test cases.
  - Then show the generated Markdown file — "ready to paste into Confluence."

Run:
  python src/use_cases/sprint_planner.py
  python src/use_cases/sprint_planner.py --limit 3
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

from src.common.data_prep import normalize_case
from src.common.validation import require_columns, require_min_rows

# ─── Config ───────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "requirements_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sprint_planner"

REQUIRED_COLUMNS = [
    "requirement_id", "raw_requirement_text", "product_area",
    "requester_role", "business_context", "delivery_window"
]

# ─── Input validation ─────────────────────────────────────────────────────────


def validate_row(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Validate a single requirement row before sending to Gemini.
    Returns (is_valid, list_of_issues).
    """
    issues = []

    text = str(row.get("raw_requirement_text", "")).strip()
    if len(text) < 30:
        issues.append("raw_requirement_text is too short (< 30 chars)")

    product_area = str(row.get("product_area", "")).strip()
    if not product_area or product_area.lower() == "nan":
        issues.append("product_area is missing")

    return len(issues) == 0, issues


# ─── Pydantic output schema ───────────────────────────────────────────────────


class Story(BaseModel):
    """A single user story with all the fields a developer team needs."""

    title: str = Field(description="Short story title.")
    description: str = Field(description="Story description in plain language.")
    acceptance_criteria: list[str] = Field(
        description="Clear, testable acceptance criteria as a list."
    )
    test_cases: list[str] = Field(
        description="Concrete test cases QA or a developer can execute."
    )
    dependencies: list[str] = Field(
        description="Likely blockers or dependencies. Empty list if none."
    )
    estimate_points: int = Field(
        ge=1, le=21,
        description="Story points as a small integer (Fibonacci-style: 1,2,3,5,8,13,21)."
    )


class SprintPlan(BaseModel):
    """Full sprint plan returned by Gemini for a single requirement."""

    epic_name: str = Field(description="Name of the parent epic.")
    business_goal: str = Field(description="Why this feature matters in one sentence.")
    stories: list[Story] = Field(
        min_length=1,
        description="List of user stories decomposed from the requirement."
    )
    demo_plan: list[str] = Field(
        description="Steps to demo this feature in a hackathon or sprint review."
    )
    risk_flags: list[str] = Field(
        description="Potential risks or assumptions that need clarification."
    )


# ─── Output validation ────────────────────────────────────────────────────────


def validate_output(plan: SprintPlan, req_id: str) -> list[str]:
    """
    Apply business rules on top of Pydantic validation.
    Returns a list of warning strings (empty = all clear).
    """
    warnings = []

    # A meaningful plan must have at least 2 stories.
    if len(plan.stories) < 2:
        warnings.append(
            f"{req_id}: only {len(plan.stories)} story generated. "
            "Requirement may be too vague — consider adding more detail."
        )

    # Every story should have at least 1 acceptance criterion.
    for story in plan.stories:
        if len(story.acceptance_criteria) == 0:
            warnings.append(
                f"{req_id}: story '{story.title}' has no acceptance criteria."
            )
        # Story points should be realistic for a 2-week sprint.
        if story.estimate_points > 13:
            warnings.append(
                f"{req_id}: story '{story.title}' has {story.estimate_points} points. "
                "Consider splitting into smaller stories."
            )

    # Total sprint points should not exceed a typical team capacity.
    total_points = sum(s.estimate_points for s in plan.stories)
    if total_points > 60:
        warnings.append(
            f"{req_id}: total estimated points={total_points}. "
            "This may be too much for a single sprint."
        )

    return warnings


# ─── Markdown writer ──────────────────────────────────────────────────────────


def write_markdown(plan: SprintPlan, req_id: str, out_path: Path) -> None:
    """
    Convert the structured sprint plan into Markdown so it can be pasted
    directly into Confluence, Notion, or GitHub wiki.
    """
    lines = [
        f"# Sprint Plan: {plan.epic_name}",
        "",
        f"**Requirement ID:** {req_id}  ",
        f"**Business goal:** {plan.business_goal}",
        "",
        "---",
        "",
        "## User Stories",
        "",
    ]
    for i, story in enumerate(plan.stories, start=1):
        lines += [
            f"### Story {i}: {story.title}",
            "",
            f"{story.description}",
            "",
            f"**Estimate:** {story.estimate_points} points",
            "",
            "**Acceptance Criteria:**",
        ]
        for ac in story.acceptance_criteria:
            lines.append(f"- {ac}")
        lines += ["", "**Test Cases:**"]
        for tc in story.test_cases:
            lines.append(f"- {tc}")
        if story.dependencies:
            lines += ["", "**Dependencies:**"]
            for dep in story.dependencies:
                lines.append(f"- {dep}")
        lines.append("")

    if plan.risk_flags:
        lines += ["## Risk Flags", ""]
        for risk in plan.risk_flags:
            lines.append(f"- {risk}")
        lines.append("")

    if plan.demo_plan:
        lines += ["## Demo Plan", ""]
        for step in plan.demo_plan:
            lines.append(f"- {step}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Prompt builder ───────────────────────────────────────────────────────────


def build_prompt(row: pd.Series) -> str:
    """
    Build a structured planning prompt that gives Gemini all available context
    so the output stories are relevant and actionable.
    """
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
- Break this down into a practical sprint plan.
- Write realistic, implementation-friendly user stories.
- Acceptance criteria must be testable statements.
- Test cases should be specific enough for a developer or QA to execute.
- Use conservative story point estimates (Fibonacci: 1,2,3,5,8,13,21).
- If something is unclear, flag it in risk_flags rather than inventing details.
- Demo plan should describe what to show in a 5-minute demo.
"""


# ─── Main processing loop ─────────────────────────────────────────────────────


def process_requirements(df: pd.DataFrame, client, limit: int) -> dict:
    """Process each requirement row through Gemini and save structured results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "sprint_plans.jsonl"
    markdown_dir = OUTPUT_DIR / "markdown"
    markdown_dir.mkdir(exist_ok=True)
    skipped_path = OUTPUT_DIR / "skipped_requirements.json"
    warnings_path = OUTPUT_DIR / "output_warnings.json"

    results, skipped, all_warnings = [], [], []
    subset = df.head(limit)
    print(f"\nProcessing {len(subset)} requirements (limit={limit})...")

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        # ── Pre-validate ─────────────────────────────────────────────────────
        is_valid, issues = validate_row(row)
        if not is_valid:
            print(f"  [SKIP] {row['requirement_id']}: {issues}")
            skipped.append({"requirement_id": row["requirement_id"], "issues": issues})
            continue

        # ── Call Gemini ──────────────────────────────────────────────────────
        prompt = build_prompt(row)
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": SprintPlan.model_json_schema(),
                },
            )
        except Exception as api_error:
            print(f"  [API ERROR] {row['requirement_id']}: {api_error}")
            skipped.append({"requirement_id": row["requirement_id"], "issues": [str(api_error)]})
            continue

        # ── Validate schema ───────────────────────────────────────────────────
        try:
            plan = SprintPlan.model_validate_json(response.text)
        except (ValidationError, ValueError) as e:
            print(f"  [SCHEMA ERROR] {row['requirement_id']}: {e}")
            skipped.append({"requirement_id": row["requirement_id"], "issues": [str(e)]})
            continue

        # ── Business warnings ─────────────────────────────────────────────────
        warnings = validate_output(plan, row["requirement_id"])
        if warnings:
            print(f"  [WARN] {warnings}")
            all_warnings.extend(warnings)

        # ── Save Markdown ─────────────────────────────────────────────────────
        md_path = markdown_dir / f"{row['requirement_id']}.md"
        write_markdown(plan, row["requirement_id"], md_path)

        # ── Collect result ────────────────────────────────────────────────────
        result_dict = {
            "requirement_id": row["requirement_id"],
            "product_area": row["product_area"],
            **plan.model_dump(),
        }
        results.append(result_dict)

        total_points = sum(s.estimate_points for s in plan.stories)
        print(f"  [{idx}/{len(subset)}] {row['requirement_id']} => "
              f"epic='{plan.epic_name}', stories={len(plan.stories)}, "
              f"total_points={total_points}")

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
        "markdown_dir": str(markdown_dir),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint planner using Gemini")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max requirements to process (default: 10)")
    args = parser.parse_args()

    print("=" * 70)
    print("USE CASE 2: REQUIREMENT-TO-SPRINT PLAN WITH GEMINI")
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

    print(f"\n[3/5] Loading clean requirements from: {CLEAN_DATA_PATH}")
    if not CLEAN_DATA_PATH.exists():
        print("ERROR: Run  python scripts/prepare_datasets.py  first.")
        sys.exit(1)
    df = pd.read_csv(CLEAN_DATA_PATH)
    require_columns(df, REQUIRED_COLUMNS, "requirements_clean")
    require_min_rows(df, 1, "requirements_clean")
    print(f"OK: Loaded {len(df)} rows. Processing first {args.limit}.")

    print("\n[4/5] Sending requirements to Gemini...")
    summary = process_requirements(df, client, args.limit)

    print("\n[5/5] SUMMARY")
    print("-" * 70)
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print("-" * 70)
    print("\nDone. Check sprint_plans.jsonl and the markdown/ folder.")


if __name__ == "__main__":
    main()
