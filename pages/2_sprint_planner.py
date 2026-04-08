"""
pages/2_sprint_planner.py
=========================
Streamlit page for Use Case 2: Requirement-to-Sprint Planner.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(page_title="Sprint Planner", page_icon="📋", layout="wide")

with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=40)
    st.title("Gemini Dev Use Cases")
    st.markdown("---")
    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
        placeholder="Paste your key here",
    )
    if api_key:
        st.session_state["gemini_api_key"] = api_key

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

# ── Gemini call ─────────────────────────────────────────────────────────────
def call_gemini(req_text: str, area: str, requester: str, context: str, window: str) -> SprintPlan:
    from google import genai
    key = st.session_state.get("gemini_api_key","")
    if not key:
        raise ValueError("No API key. Enter it in the sidebar.")
    os.environ["GEMINI_API_KEY"] = key
    client = genai.Client()
    prompt = f"""You are a senior product-minded engineering lead.

Context:
  Product area    : {area}
  Requested by    : {requester}
  Business context: {context}
  Delivery window : {window}

Raw requirement:
{req_text}

Instructions:
- Break this into a practical sprint plan with realistic user stories.
- Acceptance criteria must be testable statements.
- Test cases must be specific enough for a developer or QA to execute.
- Use Fibonacci story points (1,2,3,5,8,13,21).
- Flag unclear items in risk_flags rather than inventing details.
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": SprintPlan.model_json_schema(),
        },
    )
    return SprintPlan.model_validate_json(response.text)

# ── Render result ─────────────────────────────────────────────────────────
def render_result(plan: SprintPlan, req_id: str = "CUSTOM") -> None:
    total_pts = sum(s.estimate_points for s in plan.stories)

    # Top-level metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Epic", plan.epic_name)
    m2.metric("Stories", len(plan.stories))
    m3.metric("Total Story Points", total_pts)

    st.markdown(f"> **Business goal:** {plan.business_goal}")
    st.markdown("---")

    # Stories as expandable cards
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
                st.markdown("**🔗 Dependencies**")
                st.markdown(", ".join(story.dependencies))

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

    # Markdown download
    lines = [f"# Sprint Plan: {plan.epic_name}", "", f"**Business goal:** {plan.business_goal}", ""]
    for i, s in enumerate(plan.stories, 1):
        lines += [f"## Story {i}: {s.title} ({s.estimate_points} pts)", f"_{s.description}_", "",
                  "**Acceptance Criteria:**"]
        lines += [f"- {ac}" for ac in s.acceptance_criteria]
        lines += ["", "**Test Cases:**"] + [f"- {tc}" for tc in s.test_cases] + [""]
    md_str = "
".join(lines)
    json_str = plan.model_dump_json(indent=2)

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download Markdown (Confluence-ready)", data=md_str,
                       file_name=f"sprint_plan_{req_id}.md", mime="text/markdown")
    c2.download_button("⬇️ Download JSON", data=json_str,
                       file_name=f"sprint_plan_{req_id}.json", mime="application/json")

# ── Main ───────────────────────────────────────────────────────────────────
st.title("📋 Use Case 2: Sprint Planner")
st.caption("Raw requirement text → epic, user stories, acceptance criteria, test cases, story points.")

mode = st.radio("Input mode:", ["✍️ Type / Paste requirement", "📂 Load from CSV"], horizontal=True)

if mode == "✍️ Type / Paste requirement":
    with st.form("req_form"):
        req_text = st.text_area(
            "Requirement text",
            value="We need a document approval workflow. Employees should upload a PDF, choose an approver, and track status as Submitted, Under Review, Approved, or Rejected. Approvers should be able to comment. The submitter should receive email on approval or rejection. Admins want a dashboard showing pending approvals and average turnaround time. Keep v1 web-only.",
            height=160,
        )
        c1, c2, c3, c4 = st.columns(4)
        area      = c1.selectbox("Product area", ["document workflow","payments","inventory","analytics","identity","notifications"])
        requester = c2.selectbox("Requested by", ["product manager","client stakeholder","ops lead","engineering manager","sales engineer"])
        context   = c3.text_input("Business context", value="Needed for enterprise rollout next quarter.")
        window    = c4.selectbox("Delivery window", ["2 weeks","1 month","next quarter","TBD"])
        submitted = st.form_submit_button("🚀 Generate Sprint Plan", type="primary")

    if submitted:
        if not req_text.strip():
            st.error("Requirement text is required.")
        elif not st.session_state.get("gemini_api_key"):
            st.error("Please enter your Gemini API key in the sidebar first.")
        else:
            with st.spinner("Gemini is writing your sprint plan…"):
                try:
                    plan = call_gemini(req_text, area, requester, context, window)
                    st.success("Sprint plan ready!", icon="✅")
                    render_result(plan)
                except (ValidationError, ValueError) as e:
                    st.error(f"Validation error: {e}")
                except Exception as e:
                    st.error(f"API error: {e}")

else:
    csv_path = PROJECT_ROOT / "data" / "clean" / "requirements_clean.csv"
    if not csv_path.exists():
        st.error("Clean data not found. Run `python scripts/prepare_datasets.py` first.")
    else:
        df = pd.read_csv(csv_path)
        st.info(f"Loaded **{len(df):,} requirements** from `data/clean/requirements_clean.csv`")
        row_idx = st.slider("Select requirement row", 0, min(len(df)-1, 999), 0)
        row = df.iloc[row_idx]

        with st.expander("👁️ Preview raw row", expanded=True):
            st.markdown(f"**ID:** `{row['requirement_id']}`  |  **Area:** {row.get('product_area','—')}  |  **By:** {row.get('requester_role','—')}")
            st.text_area("Requirement text", value=str(row["raw_requirement_text"]), height=120, disabled=True, key="csv_req_preview")

        if st.button("🚀 Generate Sprint Plan", type="primary"):
            if not st.session_state.get("gemini_api_key"):
                st.error("Please enter your Gemini API key in the sidebar first.")
            else:
                with st.spinner("Gemini is writing your sprint plan…"):
                    try:
                        plan = call_gemini(
                            str(row["raw_requirement_text"]),
                            str(row.get("product_area","general")),
                            str(row.get("requester_role","product manager")),
                            str(row.get("business_context","")),
                            str(row.get("delivery_window","TBD")),
                        )
                        st.success("Sprint plan ready!", icon="✅")
                        render_result(plan, str(row["requirement_id"]))
                    except (ValidationError, ValueError) as e:
                        st.error(f"Validation error: {e}")
                    except Exception as e:
                        st.error(f"API error: {e}")
