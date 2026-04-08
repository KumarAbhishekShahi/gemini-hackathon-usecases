"""
pages/1_ticket_triage.py
========================
Streamlit page for Use Case 1: Support Ticket Triage.

Uses the shared Gemini REST client (src/common/gemini_client.py) which handles:
  - 429 retryDelay parsing from response body
  - Exponential backoff with full jitter (base=15 s, max=120 s)
  - Inter-request throttle (6.5 s gap, safe for free-tier 10 RPM)

The Streamlit spinner and status messages show live wait feedback.
"""

import json
import os
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Shared REST client — handles all 429 / retry / throttle logic.
from src.common.gemini_client import call_gemini_rest as _rest_call

st.set_page_config(page_title="Ticket Triage", page_icon="🎫", layout="wide")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg",
        width=40,
    )
    st.title("Gemini Dev Use Cases")
    st.markdown("---")
    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
        placeholder="Paste your key here",
    )
    if api_key_input:
        st.session_state["gemini_api_key"] = api_key_input
    st.markdown("---")
    st.caption("Transport: **REST (requests)** — no gRPC.")
    st.caption("Free tier: **10 RPM** — client throttles automatically.")

# ── Config ────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"

# ── Pydantic schema ────────────────────────────────────────────────────────
class TicketAnalysis(BaseModel):
    category: Literal["bug","feature_request","access_issue","billing","how_to","other"]
    priority: Literal["low","medium","high","critical"]
    likely_team: Literal["frontend","backend","qa","devops","security","support","product","unknown"]
    short_summary: str
    customer_visible_impact: str
    reproduction_steps: list[str]
    suggested_reply: str
    confidence: int = Field(ge=1, le=100)


# ── Call helper ────────────────────────────────────────────────────────────

def call_and_parse(
    subject: str, body: str,
    tier: str, app_module: str, platform: str, region: str,
    api_key: str,
    status_placeholder,
) -> TicketAnalysis:
    """
    Build prompt → call shared REST client → validate with Pydantic.

    'status_placeholder' is an st.empty() so we can update the spinner
    message with live 429 wait feedback without re-rendering the whole page.
    """

    prompt = f"""You are a senior support triage assistant for a SaaS software company.

Context:
  Customer tier : {tier}
  App module    : {app_module}
  Platform      : {platform}
  Region        : {region}

Subject: {subject}

Body:
{body}

Instructions:
- Classify using only information provided. Be conservative.
- Do not invent reproduction steps.
- Priority must reflect business impact.
- Suggested reply must be polite, professional, and actionable.
- Confidence is an integer 1-100.
"""
    # log_fn forwards Gemini client log messages to the Streamlit status area.
    def st_log(msg: str) -> None:
        # Show 429 wait messages prominently; others as debug text.
        if "429" in msg or "RATE LIMITED" in msg or "THROTTLE" in msg or "Waiting" in msg:
            status_placeholder.warning(f"⏳ {msg}", icon="⏳")
        else:
            status_placeholder.info(f"🔄 {msg}")

    raw_json = _rest_call(
        prompt  = prompt,
        api_key = api_key,
        schema  = TicketAnalysis.model_json_schema(),
        model   = GEMINI_MODEL,
        log_fn  = st_log,
    )
    return TicketAnalysis.model_validate_json(raw_json)


# ── Display helpers ────────────────────────────────────────────────────────
PRIORITY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
TEAM_ICON = {
    "frontend": "🖥️", "backend": "⚙️", "qa": "🧪", "devops": "🚀",
    "security": "🔒", "support": "💬", "product": "📦", "unknown": "❓",
}

def render_result(result: TicketAnalysis) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Priority",   f"{PRIORITY_ICON.get(result.priority, '')} {result.priority.upper()}")
    m2.metric("Category",   result.category.replace("_", " ").title())
    m3.metric("Team",       f"{TEAM_ICON.get(result.likely_team, '')} {result.likely_team}")
    m4.metric("Confidence", f"{result.confidence}/100")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📝 Summary")
        st.info(result.short_summary)

        st.markdown("#### 🧑‍💼 Customer Impact")
        st.warning(result.customer_visible_impact)

        st.markdown("#### 🔁 Reproduction Steps")
        if result.reproduction_steps:
            for i, step in enumerate(result.reproduction_steps, 1):
                st.markdown(f"{i}. {step}")
        else:
            st.caption("_Not available from the provided text._")

    with col_r:
        st.markdown("#### 💬 Suggested Reply")
        st.text_area("Copy and send this reply:", value=result.suggested_reply, height=220, key="reply_box")

    with st.expander("🛠️ Debug — Request schema sent to Gemini REST API"):
        st.code(
            json.dumps(TicketAnalysis.model_json_schema(), indent=2),
            language="json"
        )
        st.markdown("**Parsed output:**")
        st.code(result.model_dump_json(indent=2), language="json")

    st.markdown("---")
    st.download_button(
        "⬇️ Download triage_result.json",
        data=result.model_dump_json(indent=2),
        file_name="triage_result.json",
        mime="application/json",
    )


# ── Rate-limit info banner ─────────────────────────────────────────────────
st.info(
    "**Free-tier rate limit:** `gemini-2.0-flash` = **10 RPM** "
    "(1 request per 6 seconds).  "
    "The client throttles automatically and reads `retryDelay` from 429 responses. "
    "No action needed on your end.",
    icon="ℹ️",
)

# ── Main ───────────────────────────────────────────────────────────────────
st.title("🎫 Use Case 1: Support Ticket Triage")
st.caption(
    "Raw ticket text → priority, team, draft reply.  "
    "**Transport: Gemini REST API via `requests` (no gRPC) · 429-safe client.**"
)

mode = st.radio("Input mode:", ["✍️ Type / Paste ticket", "📂 Load from CSV"], horizontal=True)

if mode == "✍️ Type / Paste ticket":
    with st.form("manual_form"):
        subject_in = st.text_input("Subject / Title", value="Checkout stuck after coupon apply")
        body_in = st.text_area(
            "Ticket body",
            value=(
                "Hi team, since this morning users report that after applying SAVE20 "
                "the checkout spins forever. We reproduced it twice on mobile Chrome. "
                "One user was charged but order did not appear in order history. "
                "This started after yesterday's pricing deployment. Please check urgently."
            ),
            height=160,
        )
        c1, c2, c3, c4 = st.columns(4)
        tier_in     = c1.selectbox("Customer tier",  ["free", "pro", "enterprise"], index=1)
        app_in      = c2.selectbox("App module",     ["checkout", "billing", "identity", "orders", "documents", "analytics"])
        platform_in = c3.selectbox("Platform",       ["web", "android", "ios", "desktop"])
        region_in   = c4.selectbox("Region",         ["IN", "US", "EU", "APAC", "MEA"])
        submitted   = st.form_submit_button("🚀 Analyse with Gemini", type="primary")

    if submitted:
        key = st.session_state.get("gemini_api_key", "")
        if not subject_in.strip() or not body_in.strip():
            st.error("Subject and body are required.")
        elif not key:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            status = st.empty()   # live status placeholder for retry messages
            with st.spinner("Calling Gemini REST API…"):
                try:
                    result = call_and_parse(
                        subject_in, body_in, tier_in, app_in, platform_in, region_in,
                        key, status,
                    )
                    status.empty()
                    st.success("Analysis complete!", icon="✅")
                    render_result(result)
                except requests.HTTPError as e:
                    status.empty()
                    st.error(
                        f"HTTP error: {e}\n\n"
                        "If this is a 429 that persisted after all retries, "
                        "wait ~60 s and try again, or reduce request frequency."
                    )
                except requests.Timeout:
                    status.empty()
                    st.error("Request timed out. Check your network and retry.")
                except (ValidationError, ValueError) as e:
                    status.empty()
                    st.error(f"Validation / response error: {e}")

else:
    csv_path = PROJECT_ROOT / "data" / "clean" / "tickets_clean.csv"
    if not csv_path.exists():
        st.error("Clean data not found. Run `python scripts/prepare_datasets.py` first.")
    else:
        df = pd.read_csv(csv_path)
        st.info(f"Loaded **{len(df):,} tickets** from `data/clean/tickets_clean.csv`")
        row_idx = st.slider("Select ticket row", 0, min(len(df) - 1, 999), 0)
        row = df.iloc[row_idx]

        with st.expander("👁️ Preview raw row", expanded=True):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Ticket ID:** `{row['ticket_id']}`")
            c1.markdown(f"**App:** {row.get('app', '—')}  |  **Platform:** {row.get('platform', '—')}")
            c2.markdown(f"**Subject:** {row['subject']}")
            c2.markdown(f"**Tier:** {row.get('customer_tier', '—')}  |  **Region:** {row.get('region', '—')}")
            st.text_area("Body", value=str(row["body"]), height=120, disabled=True, key="csv_body_preview")

        if st.button("🚀 Analyse this ticket with Gemini", type="primary"):
            key = st.session_state.get("gemini_api_key", "")
            if not key:
                st.error("Enter your Gemini API key in the sidebar first.")
            else:
                status = st.empty()
                with st.spinner("Calling Gemini REST API…"):
                    try:
                        result = call_and_parse(
                            str(row["subject"]), str(row["body"]),
                            str(row.get("customer_tier", "pro")),
                            str(row.get("app", "orders")),
                            str(row.get("platform", "web")),
                            str(row.get("region", "IN")),
                            key, status,
                        )
                        status.empty()
                        st.success("Analysis complete!", icon="✅")
                        render_result(result)
                    except requests.HTTPError as e:
                        status.empty()
                        st.error(f"HTTP error: {e}")
                    except requests.Timeout:
                        status.empty()
                        st.error("Request timed out.")
                    except (ValidationError, ValueError) as e:
                        status.empty()
                        st.error(f"Validation / response error: {e}")
