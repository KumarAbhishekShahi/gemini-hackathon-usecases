"""
pages/3_log_explainer.py
========================
Streamlit page for Use Case 3: Incident Log Explainer.
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

st.set_page_config(page_title="Log Explainer", page_icon="🔍", layout="wide")

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
class IncidentAnalysis(BaseModel):
    affected_component: str
    probable_root_cause: str
    confidence: int = Field(ge=1, le=100)
    what_happened: str
    evidence_lines: list[str]
    immediate_actions: list[str] = Field(min_length=1)
    code_fix_suggestions: list[str]
    monitoring_checks: list[str]
    incident_severity: str = Field(pattern=r"^(sev1|sev2|sev3|sev4)$")

# ── Call helper ─────────────────────────────────────────────────────────────
def call_and_parse(
    log_text: str, service: str, environment: str, region: str,
    api_key: str, status_placeholder,
) -> IncidentAnalysis:
    prompt = f"""You are a senior SRE and backend developer.

Context:
  Service     : {service}
  Environment : {environment}
  Region      : {region}

Log chunk:
{log_text}

Instructions:
- Base analysis ONLY on the log content above. Do not invent details.
- Evidence lines must quote or closely paraphrase actual log content.
- Immediate actions must be safe (no destructive ops without confirmation).
- incident_severity: sev1=full outage, sev2=major degradation, sev3=minor, sev4=informational.
- If log is incomplete, say so and lower your confidence.
"""
    def st_log(msg: str) -> None:
        if any(k in msg for k in ("429", "503", "RATE", "FALLBACK", "THROTTLE", "Waiting", "overloaded")):
            status_placeholder.warning(f"⏳ {msg}", icon="⏳")
        else:
            status_placeholder.info(f"🔄 {msg}")

    raw = _rest_call(
        prompt=prompt, api_key=api_key,
        schema=IncidentAnalysis.model_json_schema(),
        model=GEMINI_MODEL, log_fn=st_log,
    )
    return IncidentAnalysis.model_validate_json(raw)

# ── Render ─────────────────────────────────────────────────────────────────
SEV_LABELS = {
    "sev1": "🔴 SEV1 — Critical Outage",
    "sev2": "🟠 SEV2 — Major Degradation",
    "sev3": "🟡 SEV3 — Minor Issue",
    "sev4": "🟢 SEV4 — Informational",
}

def render_result(analysis: IncidentAnalysis, incident_id: str = "CUSTOM") -> None:
    sev_label = SEV_LABELS.get(analysis.incident_severity, analysis.incident_severity)
    if analysis.incident_severity == "sev1":
        st.error(f"**{sev_label}** — Component: `{analysis.affected_component}`", icon="🚨")
    elif analysis.incident_severity == "sev2":
        st.warning(f"**{sev_label}** — Component: `{analysis.affected_component}`", icon="⚠️")
    else:
        st.info(f"**{sev_label}** — Component: `{analysis.affected_component}`", icon="ℹ️")

    m1, m2 = st.columns([3, 1])
    m1.markdown(f"**🎯 Probable Root Cause:** {analysis.probable_root_cause}")
    m2.metric("Confidence", f"{analysis.confidence}/100")

    st.markdown("---")
    st.markdown(f"**📖 What Happened:** {analysis.what_happened}")
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🔎 Evidence Lines")
        if analysis.evidence_lines:
            for ev in analysis.evidence_lines:
                st.code(ev, language="text")
        else:
            st.caption("_No specific evidence lines found._")
        st.markdown("### 🛑 Immediate Actions")
        for i, action in enumerate(analysis.immediate_actions, 1):
            st.markdown(f"{i}. {action}")

    with col_r:
        st.markdown("### 🛠️ Code Fix Suggestions")
        if analysis.code_fix_suggestions:
            for fix in analysis.code_fix_suggestions:
                st.markdown(f"- {fix}")
        else:
            st.caption("_No code fixes identified._")
        st.markdown("### 📊 Monitoring Checks")
        if analysis.monitoring_checks:
            for check in analysis.monitoring_checks:
                st.markdown(f"- {check}")

    runbook = "\n".join([
        f"# Incident Runbook: {incident_id}", "",
        f"**Severity:** `{analysis.incident_severity}`",
        f"**Component:** {analysis.affected_component}",
        f"**Confidence:** {analysis.confidence}/100", "",
        "## What Happened", "", analysis.what_happened, "",
        "## Probable Root Cause", "", analysis.probable_root_cause, "",
        "## Evidence", "",
    ] + [f"- `{ev}`" for ev in analysis.evidence_lines] + [
        "", "## Immediate Actions", "",
    ] + [f"{i}. {a}" for i, a in enumerate(analysis.immediate_actions, 1)] + [
        "", "## Code Fix Suggestions", "",
    ] + [f"- {f}" for f in analysis.code_fix_suggestions] + [
        "", "## Monitoring Checks", "",
    ] + [f"- {c}" for c in analysis.monitoring_checks])

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download Runbook (Markdown)", data=runbook,
                       file_name=f"runbook_{incident_id}.md", mime="text/markdown")
    c2.download_button("⬇️ Download JSON", data=analysis.model_dump_json(indent=2),
                       file_name=f"incident_{incident_id}.json", mime="application/json")

# ── Page ───────────────────────────────────────────────────────────────────
st.info(
    "**503 auto-fallback active:** if `gemini-2.5-flash` is overloaded the client "
    "automatically retries with `gemini-2.0-flash` → `gemini-1.5-flash`.",
    icon="ℹ️",
)
st.title("🔍 Use Case 3: Incident Log Explainer")
st.caption("Logs/stack trace → root-cause, evidence, actions, downloadable runbook.")

mode = st.radio("Input mode:", ["✍️ Paste log / stack trace", "📂 Load from CSV"], horizontal=True)

if mode == "✍️ Paste log / stack trace":
    with st.form("log_form"):
        log_text = st.text_area(
            "Log chunk / stack trace",
            value="""2026-04-08 09:15:01 INFO  order-service Starting request POST /api/orders
2026-04-08 09:15:01 INFO  pricing-client Calling pricing service for cartId=88421
2026-04-08 09:15:03 WARN  pricing-client Timeout while calling pricing service after 2000ms
2026-04-08 09:15:03 ERROR order-service Failed to create order
java.lang.RuntimeException: pricing lookup failed
    at com.acme.order.PricingAdapter.fetchPrice(PricingAdapter.java:84)
2026-04-08 09:15:03 INFO  retry-handler Retrying pricing call attempt=2
2026-04-08 09:15:05 WARN  pricing-client Timeout while calling pricing service after 2000ms
2026-04-08 09:15:05 ERROR payment-api Order creation aborted before payment authorization
2026-04-08 09:15:05 INFO  gateway Response status=500 path=/api/orders""",
            height=220,
        )
        c1, c2, c3 = st.columns(3)
        service     = c1.text_input("Service name", value="order-service")
        environment = c2.selectbox("Environment", ["prod","qa","dev"])
        region      = c3.selectbox("Region", ["IN","US","EU","APAC","MEA"])
        submitted   = st.form_submit_button("🚀 Explain with Gemini", type="primary")

    if submitted:
        key = st.session_state.get("gemini_api_key","")
        if len(log_text.strip()) < 40:
            st.error("Log text is too short.")
        elif not key:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            status = st.empty()
            with st.spinner("Calling Gemini REST API…"):
                try:
                    analysis = call_and_parse(log_text, service, environment, region, key, status)
                    status.empty()
                    st.success("Analysis complete!", icon="✅")
                    render_result(analysis)
                except requests.HTTPError as e:
                    status.empty()
                    st.error(f"HTTP error: {e}")
                except (ValidationError, ValueError) as e:
                    status.empty()
                    st.error(f"Error: {e}")

else:
    csv_path = PROJECT_ROOT / "data" / "clean" / "logs_clean.csv"
    if not csv_path.exists():
        st.error("Run `python scripts/prepare_datasets.py` first.")
    else:
        df = pd.read_csv(csv_path)
        env_filter = st.selectbox("Filter by environment:", ["all","prod","qa","dev"])
        if env_filter != "all":
            df = df[df["environment"] == env_filter].reset_index(drop=True)
        st.info(f"Showing **{len(df):,} records**")
        row_idx = st.slider("Select incident row", 0, min(len(df)-1, 999), 0)
        row = df.iloc[row_idx]
        with st.expander("👁️ Preview raw log chunk", expanded=True):
            st.markdown(f"**ID:** `{row['incident_id']}`  |  **Service:** {row.get('service_name','—')}  |  **Env:** {row.get('environment','—')}")
            st.code(str(row["log_chunk"]), language="text")
        if st.button("🚀 Explain this incident with Gemini", type="primary"):
            key = st.session_state.get("gemini_api_key","")
            if not key:
                st.error("Enter your Gemini API key in the sidebar first.")
            else:
                status = st.empty()
                with st.spinner("Calling Gemini REST API…"):
                    try:
                        analysis = call_and_parse(
                            str(row["log_chunk"]),
                            str(row.get("service_name","unknown")),
                            str(row.get("environment","prod")),
                            str(row.get("region","IN")),
                            key, status,
                        )
                        status.empty()
                        st.success("Analysis complete!", icon="✅")
                        render_result(analysis, str(row["incident_id"]))
                    except requests.HTTPError as e:
                        status.empty()
                        st.error(f"HTTP error: {e}")
                    except (ValidationError, ValueError) as e:
                        status.empty()
                        st.error(f"Error: {e}")
