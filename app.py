"""
app.py — Main Streamlit entry point
====================================
Run:  streamlit run app.py

This is the Home page. Navigation to each use case happens via the Streamlit
sidebar. Each use case is a separate file inside the /pages folder, which
Streamlit automatically picks up and shows in the sidebar.

Sidebar also hosts the Gemini API key input so it persists across all pages.
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Gemini Dev Use Cases",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: API key (shared across all pages via session_state) ───────────
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=40)
    st.title("Gemini Dev Use Cases")
    st.markdown("---")
    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
        placeholder="Paste your key here",
        help="Get a free key at https://aistudio.google.com",
    )
    if api_key:
        st.session_state["gemini_api_key"] = api_key
        st.success("API key saved for this session.", icon="✅")
    else:
        st.warning("Enter your API key to run demos.", icon="⚠️")
    st.markdown("---")
    st.caption("All 3 demos use Gemini structured JSON output + Pydantic validation.")

# ── Home page content ──────────────────────────────────────────────────────
st.title("🤖 Gemini Developer Use Cases")
st.subheader("Hackathon Starter Kit — 3 Real-World Demos")
st.markdown("""
> **Paste raw text → Gemini analyses it → Get validated structured JSON**  
> Every demo shows the *before* (messy input) and *after* (clean structured output) side by side.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎫 Ticket Triage
    Raw support emails and bug reports → priority, team routing, and a draft reply.

    **Pain point:** Support inbox filled with unstructured text.  
    **Output:** `priority`, `likely_team`, `category`, `suggested_reply`, `confidence`
    """)
    st.page_link("pages/1_ticket_triage.py", label="Open Ticket Triage →", icon="🎫")

with col2:
    st.markdown("""
    ### 📋 Sprint Planner
    Vague PM or client requirement text → user stories, acceptance criteria, test cases, story points.

    **Pain point:** Manual requirement decomposition wastes developer time.  
    **Output:** `epic`, `stories[]`, `acceptance_criteria[]`, `test_cases[]`, `estimate_points`
    """)
    st.page_link("pages/2_sprint_planner.py", label="Open Sprint Planner →", icon="📋")

with col3:
    st.markdown("""
    ### 🔍 Log Explainer
    Noisy logs and stack traces → root-cause hypothesis, evidence, actions, and a runbook.

    **Pain point:** On-call engineers parse logs manually during incidents.  
    **Output:** `probable_root_cause`, `evidence_lines[]`, `immediate_actions[]`, `incident_severity`
    """)
    st.page_link("pages/3_log_explainer.py", label="Open Log Explainer →", icon="🔍")

st.markdown("---")
st.markdown("""
### How it works

```
Raw text (ticket / requirement / log)
        │
        ▼
  Clean & validate input
        │
        ▼
  Gemini API  (response_mime_type=application/json + Pydantic schema)
        │
        ▼
  Validate output  (Pydantic schema + business rules)
        │
        ▼
  Display structured result + download JSON / Markdown
```
""")

with st.expander("📁 About the dataset"):
    st.markdown("""
    The **data/** folder contains **1,200 synthetic rows** per use case:
    - `data/raw/` — Messy data with noise, casing issues, blanks, and duplicates.
    - `data/clean/` — Cleaned output of `scripts/prepare_datasets.py`.

    The demos can use either **typed text** (live input) or **rows from the clean CSV**
    so you can batch-process and show real volume during a hackathon presentation.
    """)
