"""
pages/1_ticket_triage.py
========================
Streamlit page for Use Case 1: Support Ticket Triage.

Uses the Gemini REST API via the 'requests' library — no gRPC, no google-genai SDK.

Why REST over gRPC?
  - No binary dependencies (grpcio) — installs cleanly everywhere.
  - Plain HTTP/JSON — trivial to inspect, log, or proxy.
  - Works the same in Streamlit Cloud, local laptops, and CI pipelines.
  - Structured output is controlled via 'responseSchema' in generationConfig.

Modes:
  - Manual input : Paste any ticket text and analyse it instantly.
  - CSV batch    : Pick a row from tickets_clean.csv and analyse it.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

# ── Path setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ticket Triage", page_icon="🎫", layout="wide")

# ── Sidebar: persist API key ───────────────────────────────────────────────
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

# ── REST config ────────────────────────────────────────────────────────────
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL     = "gemini-2.0-flash"
GEMINI_ENDPOINT  = f"{GEMINI_REST_BASE}/models/{GEMINI_MODEL}:generateContent"
MAX_RETRIES      = 3
RETRY_BACKOFF_S  = 2

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


# ── REST caller ────────────────────────────────────────────────────────────

def build_request_body(prompt: str) -> dict:
    """
    Build the JSON payload for the Gemini generateContent REST endpoint.

    generationConfig.responseSchema constrains the model to only return
    JSON that matches our TicketAnalysis Pydantic model.
    """
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            # Ask Gemini to return JSON, not prose.
            "responseMimeType": "application/json",
            # Pass the full JSON Schema so Gemini knows every field name,
            # type, and allowed enum value.
            "responseSchema": TicketAnalysis.model_json_schema(),
        },
    }


def call_gemini_rest(prompt: str, api_key: str) -> str:
    """
    Send the prompt to Gemini via HTTP POST and return the raw JSON string.

    The Gemini REST response looks like:
    {
      "candidates": [
        {
          "content": {
            "parts": [{ "text": "<json here>" }]
          }
        }
      ]
    }

    We navigate that structure and return the inner text.

    Raises:
        requests.HTTPError : For unrecoverable HTTP errors.
        ValueError         : For unexpected response shapes.
    """
    # API key goes in the query string — no OAuth, no bearer token needed.
    url = f"{GEMINI_ENDPOINT}?key={api_key}"

    headers = {
        "Content-Type": "application/json",
        "Accept":       "application/json",
    }

    body = build_request_body(prompt)

    for attempt in range(1, MAX_RETRIES + 1):
        # Make the HTTP POST request.
        response = requests.post(
            url,
            headers=headers,
            json=body,         # requests auto-serialises dict → JSON body
            timeout=(10, 60),  # (connect_timeout, read_timeout) in seconds
        )

        # Retry on rate-limit (429) or server overload (503).
        if response.status_code in (429, 503) and attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_S * (2 ** (attempt - 1))
            time.sleep(wait)
            continue

        # Raise for 4xx/5xx that are not retriable.
        response.raise_for_status()

        # Navigate the nested response envelope to get the generated text.
        try:
            generated_text = (
                response.json()["candidates"][0]["content"]["parts"][0]["text"]
            )
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected Gemini response structure: {e}. "
                f"Raw: {json.dumps(response.json(), indent=2)[:500]}"
            )

        return generated_text

    raise requests.HTTPError(f"All {MAX_RETRIES} retry attempts failed.")


def call_and_parse(
    subject: str,
    body: str,
    tier: str,
    app_module: str,
    platform: str,
    region: str,
    api_key: str,
) -> TicketAnalysis:
    """
    Build prompt → call REST API → validate with Pydantic → return result.
    Raises ValueError or ValidationError on failure.
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
- Classify the ticket accurately using only the information provided.
- Be conservative. Do not invent reproduction steps.
- Priority must reflect business impact.
- Suggested reply must be polite, professional, and actionable.
- Confidence is an integer 1-100.
"""
    raw_json = call_gemini_rest(prompt, api_key)
    return TicketAnalysis.model_validate_json(raw_json)


# ── Display helpers ────────────────────────────────────────────────────────
PRIORITY_ICON = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢"}
TEAM_ICON     = {
    "frontend":"🖥️","backend":"⚙️","qa":"🧪","devops":"🚀",
    "security":"🔒","support":"💬","product":"📦","unknown":"❓",
}

def render_result(result: TicketAnalysis) -> None:
    """Display the structured triage result as a rich Streamlit layout."""

    # ── Top metrics ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Priority",   f"{PRIORITY_ICON.get(result.priority,'')} {result.priority.upper()}")
    m2.metric("Category",   result.category.replace("_"," ").title())
    m3.metric("Team",       f"{TEAM_ICON.get(result.likely_team,'')} {result.likely_team}")
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
        st.text_area(
            "Copy and send this reply:",
            value=result.suggested_reply,
            height=220,
            key="reply_box",
        )

    # ── Debug expander: show raw HTTP request body ───────────────────────────
    with st.expander("🛠️ Debug — Raw request & response JSON"):
        st.markdown("**Request payload sent to Gemini REST API:**")
        sample_body = build_request_body("<prompt was sent here — see above>")
        st.code(json.dumps(sample_body, indent=2), language="json")
        st.markdown("**Response parsed into:**")
        st.code(result.model_dump_json(indent=2), language="json")

    st.markdown("---")
    st.download_button(
        "⬇️ Download triage_result.json",
        data=result.model_dump_json(indent=2),
        file_name="triage_result.json",
        mime="application/json",
    )


# ── Main page ──────────────────────────────────────────────────────────────
st.title("🎫 Use Case 1: Support Ticket Triage")
st.caption(
    "Raw ticket text → priority, team routing, and a draft reply.  "
    "**Transport: Gemini REST API via `requests` (no gRPC).**"
)

# ── Input mode ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Input mode:", ["✍️ Type / Paste ticket", "📂 Load from CSV"], horizontal=True
)

if mode == "✍️ Type / Paste ticket":
    with st.form("manual_form"):
        subject_in = st.text_input(
            "Subject / Title",
            value="Checkout stuck after coupon apply",
        )
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
        tier_in    = c1.selectbox("Customer tier", ["free","pro","enterprise"], index=1)
        app_in     = c2.selectbox("App module", ["checkout","billing","identity","orders","documents","analytics"])
        platform_in = c3.selectbox("Platform", ["web","android","ios","desktop"])
        region_in  = c4.selectbox("Region", ["IN","US","EU","APAC","MEA"])
        submitted  = st.form_submit_button("🚀 Analyse with Gemini", type="primary")

    if submitted:
        key = st.session_state.get("gemini_api_key","")
        if not subject_in.strip() or not body_in.strip():
            st.error("Subject and body are required.")
        elif not key:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            with st.spinner("Calling Gemini REST API…"):
                try:
                    result = call_and_parse(
                        subject_in, body_in, tier_in, app_in, platform_in, region_in, key
                    )
                    st.success("Analysis complete!", icon="✅")
                    render_result(result)
                except requests.HTTPError as e:
                    st.error(f"HTTP error calling Gemini: {e}")
                except requests.Timeout:
                    st.error("Request timed out. Check your network and try again.")
                except (ValidationError, ValueError) as e:
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
            c1.markdown(f"**App:** {row.get('app','—')}  |  **Platform:** {row.get('platform','—')}")
            c2.markdown(f"**Subject:** {row['subject']}")
            c2.markdown(f"**Tier:** {row.get('customer_tier','—')}  |  **Region:** {row.get('region','—')}")
            st.text_area(
                "Body", value=str(row["body"]), height=120,
                disabled=True, key="csv_body_preview"
            )

        if st.button("🚀 Analyse this ticket with Gemini", type="primary"):
            key = st.session_state.get("gemini_api_key","")
            if not key:
                st.error("Enter your Gemini API key in the sidebar first.")
            else:
                with st.spinner("Calling Gemini REST API…"):
                    try:
                        result = call_and_parse(
                            str(row["subject"]),
                            str(row["body"]),
                            str(row.get("customer_tier","pro")),
                            str(row.get("app","orders")),
                            str(row.get("platform","web")),
                            str(row.get("region","IN")),
                            key,
                        )
                        st.success("Analysis complete!", icon="✅")
                        render_result(result)
                    except requests.HTTPError as e:
                        st.error(f"HTTP error calling Gemini: {e}")
                    except requests.Timeout:
                        st.error("Request timed out. Check your network and try again.")
                    except (ValidationError, ValueError) as e:
                        st.error(f"Validation / response error: {e}")
