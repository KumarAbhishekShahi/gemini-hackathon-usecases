"""
pages/2_sprint_planner.py
=========================
Streamlit page for Use Case 2: Requirement-to-Sprint Planner.
Uses shared Gemini REST client — handles 429 + 503 with auto-fallback.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.gemini_client import call_gemini_rest as _rest_call

st.set_page_config(page_title="Sprint Planner", page_icon="📋", layout="wide")

with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=40)
    st.title("Gemini Dev Use Cases")
    st.markdown("---")
    api_key_input = st.text_input(
        "🔑 Gemini API Key", type="password",
        value=st.session_state.get("gemini_api_key", ""),
        placeholder="Paste your key here",
    )
    if api_key_input:
        st.session_state["gemini_api_key"] = api_key_input
    st.markdown("---")
    st.caption("Transport: **REST (requests)** — no gRPC.")
    st.caption("Auto-fallback: 2.5-flash → 2.0-flash → 1.5-flash on 503.")

GEMINI_MODEL = "gemini-2.5-flash"

# ── Schema ─────────────────────────────────────────────────────────────────
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

# ── Call helper ─────────────────────────────────────────────────────────────
def call_and_parse(
    req_text: str, area: str, requester: str, context: str, window: str,
    api_key: str, status_placeholder,
) -> SprintPlan:
    prompt = f"""You are a senior product-minded engineering lead.

Context:
  Product area    : {area}
  Requested by    : {requester}
  Business context: {context}
  Delivery window : {window}

Raw requirement:
{req_text}

Instructions:
- Break into realistic sprint stories with acceptance criteria and test cases.
- Use Fibonacci story points (1,2,3,5,8,13,21).
- Flag unclear items in risk_flags rather than inventing details.
"""
    def st_log(msg: str) -> None:
        if any(k in msg for k in ("429", "503", "RATE", "FALLBACK", "THROTTLE", "Waiting", "overloaded")):
            status_placeholder.warning(f"⏳ {msg}", icon="⏳")
        else:
            status_placeholder.info(f"🔄 {msg}")

    raw = _rest_call(
        prompt=prompt, api_key=api_key,
        schema=SprintPlan.model_json_schema(),
        model=GEMINI_MODEL, log_fn=st_log,
    )
    return SprintPlan.model_validate_json(raw)

# ── Render ─────────────────────────────────────────────────────────────────
def render_result(plan: SprintPlan, req_id: str = "CUSTOM") -> None:
    total_pts = sum(s.estimate_points for s in plan.stories)
    m1, m2, m3 = st.columns(3)
    m1.metric("Epic", plan.epic_name)
    m2.metric("Stories", len(plan.stories))
    m3.metric("Total Points", total_pts)
    st.markdown(f"> **Business goal:** {plan.business_goal}")
    st.markdown("---")

    st.markdown("### 📖 User Stories")
    POINT_COLORS = {1:"🟢",2:"🟢",3:"🟡",5:"🟡",8:"🟠",13:"🔴",21:"🔴"}
    for i, story in enumerate(plan.stories, 1):
        color = POINT_COLORS.get(story.estimate_points, "⚪")
        with st.expander(f"{color} Story {i}: {story.title}  —  **{story.estimate_points} pts**", expanded=(i==1)):
            st.markdown(f"_{story.description}_")
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**✅ Acceptance Criteria**")
                for ac in story.acceptance_criteria:
                    st.markdown(f"- {ac}")
            with col_r:
                st.markdown("**🧪 Test Cases**")
                for tc in story.test_cases:
                    st.markdown(f"- {tc}")
            if story.dependencies:
                st.markdown(f"**🔗 Dependencies:** {', '.join(story.dependencies)}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if plan.risk_flags:
            st.markdown("### ⚠️ Risk Flags")
            for r in plan.risk_flags:
                st.warning(r, icon="⚠️")
    with col_b:
        if plan.demo_plan:
            st.markdown("### 🎤 Demo Plan")
            for i, step in enumerate(plan.demo_plan, 1):
                st.markdown(f"{i}. {step}")

    lines = [f"# Sprint Plan: {plan.epic_name}", "", f"**Business goal:** {plan.business_goal}", ""]
    for i, s in enumerate(plan.stories, 1):
        lines += [f"## Story {i}: {s.title} ({s.estimate_points} pts)", f"_{s.description}_", "",
                  "**Acceptance Criteria:**"] + [f"- {ac}" for ac in s.acceptance_criteria]
        lines += ["", "**Test Cases:**"] + [f"- {tc}" for tc in s.test_cases] + [""]
    md_str   = "\n".join(lines)
    json_str = plan.model_dump_json(indent=2)
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download Markdown", data=md_str,
                       file_name=f"sprint_plan_{req_id}.md", mime="text/markdown")
    c2.download_button("⬇️ Download JSON", data=json_str,
                       file_name=f"sprint_plan_{req_id}.json", mime="application/json")

# ── Page ───────────────────────────────────────────────────────────────────
st.info(
    "**503 auto-fallback active:** if `gemini-2.5-flash` is overloaded the client "
    "automatically retries with `gemini-2.0-flash` → `gemini-1.5-flash`.",
    icon="ℹ️",
)
st.title("📋 Use Case 2: Sprint Planner")
st.caption("Raw requirement → epic, user stories, acceptance criteria, test cases, story points.")

mode = st.radio("Input mode:", ["✍️ Type / Paste requirement", "📂 Load from CSV"], horizontal=True)

if mode == "✍️ Type / Paste requirement":
    with st.form("req_form"):
        req_text = st.text_area(
            "Requirement text",
            value="We need a document approval workflow. Employees upload a PDF, choose an approver, track status (Submitted/Under Review/Approved/Rejected). Approvers can comment. Submitter gets email on decision. Admins see pending approvals dashboard with avg turnaround. Web-only v1.",
            height=160,
        )
        c1, c2, c3, c4 = st.columns(4)
        area      = c1.selectbox("Product area",     ["document workflow","payments","inventory","analytics","identity","notifications"])
        requester = c2.selectbox("Requested by",     ["product manager","client stakeholder","ops lead","engineering manager"])
        context   = c3.text_input("Business context", value="Needed for enterprise rollout next quarter.")
        window    = c4.selectbox("Delivery window",  ["2 weeks","1 month","next quarter","TBD"])
        submitted = st.form_submit_button("🚀 Generate Sprint Plan", type="primary")

    if submitted:
        key = st.session_state.get("gemini_api_key","")
        if not req_text.strip():
            st.error("Requirement text is required.")
        elif not key:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            status = st.empty()
            with st.spinner("Calling Gemini REST API…"):
                try:
                    plan = call_and_parse(req_text, area, requester, context, window, key, status)
                    status.empty()
                    st.success("Sprint plan ready!", icon="✅")
                    render_result(plan)
                except requests.HTTPError as e:
                    status.empty()
                    st.error(f"HTTP error: {e}")
                except (ValidationError, ValueError) as e:
                    status.empty()
                    st.error(f"Error: {e}")

else:
    csv_path = PROJECT_ROOT / "data" / "clean" / "requirements_clean.csv"
    if not csv_path.exists():
        st.error("Run `python scripts/prepare_datasets.py` first.")
    else:
        df = pd.read_csv(csv_path)
        st.info(f"Loaded **{len(df):,} requirements**")
        row_idx = st.slider("Select row", 0, min(len(df)-1, 999), 0)
        row = df.iloc[row_idx]
        with st.expander("👁️ Preview", expanded=True):
            st.markdown(f"**ID:** `{row['requirement_id']}`  |  **Area:** {row.get('product_area','—')}")
            st.text_area("Requirement", value=str(row["raw_requirement_text"]), height=100, disabled=True, key="req_prev")
        if st.button("🚀 Generate Sprint Plan", type="primary"):
            key = st.session_state.get("gemini_api_key","")
            if not key:
                st.error("Enter your Gemini API key in the sidebar first.")
            else:
                status = st.empty()
                with st.spinner("Calling Gemini REST API…"):
                    try:
                        plan = call_and_parse(
                            str(row["raw_requirement_text"]),
                            str(row.get("product_area","general")),
                            str(row.get("requester_role","product manager")),
                            str(row.get("business_context","")),
                            str(row.get("delivery_window","TBD")),
                            key, status,
                        )
                        status.empty()
                        st.success("Sprint plan ready!", icon="✅")
                        render_result(plan, str(row["requirement_id"]))
                    except requests.HTTPError as e:
                        status.empty()
                        st.error(f"HTTP error: {e}")
                    except (ValidationError, ValueError) as e:
                        status.empty()
                        st.error(f"Error: {e}")
