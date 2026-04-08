"""
src/common/gemini_client.py
============================
Shared Gemini REST client with production-grade 429 handling.

Why a shared module?
  Both the CLI scripts and the Streamlit pages call the same REST endpoint.
  Centralising retry/backoff/throttle logic here means one fix applies everywhere.

Root cause of 429 RESOURCE_EXHAUSTED:
  The free-tier Gemini API enforces a rolling-60-second window:
    gemini-2.0-flash : 10 RPM  (= 1 request every 6 seconds minimum)
    gemini-2.5-flash : 10 RPM
    gemini-2.5-pro   :  5 RPM
  The previous code retried at 2s → 4s, which is far too short to clear the
  60-second rate-limit window.

Three-layer fix implemented here:
  Layer 1 — Parse retryDelay from Gemini's 429 response body
    Gemini returns a structured error like:
      {"error": {"details": [{"retryDelay": "30s", "@type": "...RetryInfo"}]}}
    We read that value and wait AT LEAST that long before retrying.

  Layer 2 — Exponential backoff with full jitter
    wait = min(BASE * 2^attempt, MAX_WAIT) + random(0, JITTER)
    Jitter prevents all parallel callers from retrying simultaneously
    (thundering-herd problem).

  Layer 3 — Inter-request throttle (MIN_INTERVAL)
    A module-level timestamp ensures we never fire two requests within
    MIN_INTERVAL seconds of each other, regardless of retries.
    For 10 RPM the interval is set to 6.5 s (slightly above 6 s for safety).
"""

from __future__ import annotations

import json
import random
import sys
import time
from typing import Callable

import requests

# ── Rate-limit constants (tuned for free-tier gemini-2.0-flash) ───────────────
# 10 RPM = 1 request per 6 s.  We use 6.5 s to add a small safety margin.
MIN_INTERVAL   : float = 6.5    # minimum seconds between consecutive requests
BASE_BACKOFF_S : float = 15.0   # starting wait on first retry after 429
MAX_BACKOFF_S  : float = 120.0  # cap so we never wait more than 2 minutes
JITTER_S       : float = 3.0    # random extra seconds to avoid thundering herd
MAX_RETRIES    : int   = 5      # total retry attempts before giving up

# ── Gemini REST base ──────────────────────────────────────────────────────────
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"

# ── Module-level throttle: timestamp of the last outgoing request ─────────────
# This persists for the lifetime of the process (or Streamlit session),
# so every call through this module respects the 10 RPM limit automatically.
_last_request_ts: float = 0.0


def _parse_retry_delay(response: requests.Response) -> float | None:
    """
    Extract the retryDelay field Gemini embeds inside 429 response bodies.

    Gemini error body shape:
    {
      "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [
          {
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "30s"
          }
        ]
      }
    }

    Returns the delay in seconds as a float, or None if not present.
    """
    try:
        details = response.json().get("error", {}).get("details", [])
        for item in details:
            delay_str = item.get("retryDelay", "")
            if delay_str.endswith("s"):
                return float(delay_str[:-1])
    except (ValueError, KeyError, AttributeError):
        pass
    return None


def _parse_retry_after_header(response: requests.Response) -> float | None:
    """
    Read the HTTP Retry-After header that some Google services send.
    Value can be an integer number of seconds or an HTTP-date string.
    We only handle the integer form.
    """
    header_val = response.headers.get("Retry-After", "")
    try:
        return float(header_val)
    except (ValueError, TypeError):
        return None


def _throttle(log_fn: Callable[[str], None] | None = None) -> None:
    """
    Enforce MIN_INTERVAL between requests by sleeping if needed.
    Called BEFORE every outgoing HTTP request.
    """
    global _last_request_ts
    now  = time.monotonic()
    wait = MIN_INTERVAL - (now - _last_request_ts)
    if wait > 0:
        msg = f"[THROTTLE] Waiting {wait:.1f}s to respect 10 RPM limit…"
        if log_fn:
            log_fn(msg)
        else:
            print(msg)
        time.sleep(wait)
    _last_request_ts = time.monotonic()


def call_gemini_rest(
    prompt:         str,
    api_key:        str,
    schema:         dict,
    model:          str  = "gemini-2.0-flash",
    log_fn:         Callable[[str], None] | None = None,
    progress_fn:    Callable[[str, int], None]   | None = None,
) -> str:
    """
    Call the Gemini generateContent REST endpoint and return the raw JSON string.

    Parameters
    ----------
    prompt      : The full prompt string to send.
    api_key     : Gemini API key (from GEMINI_API_KEY env var or Streamlit state).
    schema      : Pydantic model's .model_json_schema() — passed as responseSchema.
    model       : Gemini model name (default: gemini-2.0-flash).
    log_fn      : Optional callable(msg: str) for custom logging (e.g. st.info).
    progress_fn : Optional callable(msg: str, pct: int) for progress bars.

    Returns
    -------
    Raw JSON string from Gemini (to be validated by your Pydantic model).

    Raises
    ------
    requests.HTTPError : If all retries are exhausted.
    ValueError         : If the response structure is unexpected.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(f"    {msg}")

    # Full endpoint URL — API key in query param, no OAuth needed.
    url = f"{GEMINI_REST_BASE}/models/{model}:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json",
        "Accept":       "application/json",
    }

    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            # Tell Gemini: return structured JSON, not prose.
            "responseMimeType": "application/json",
            # Pass the full Pydantic JSON Schema so Gemini constrains
            # its output to exactly the fields and types you defined.
            "responseSchema": schema,
        },
    }

    for attempt in range(1, MAX_RETRIES + 2):  # +2 so last attempt fires, then we raise
        # ── Layer 3: Throttle ────────────────────────────────────────────────
        # Always enforce the min-interval between requests,
        # whether this is the first call or a retry.
        _throttle(log_fn=_log)

        _log(f"[HTTP] POST …/{model}:generateContent (attempt {attempt}/{MAX_RETRIES + 1})")
        if progress_fn:
            pct = int(100 * attempt / (MAX_RETRIES + 1))
            progress_fn(f"Attempt {attempt} / {MAX_RETRIES + 1}", pct)

        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=(10, 60),
        )

        _log(f"[HTTP] Status: {response.status_code}")

        # ── 429: Rate limited ────────────────────────────────────────────────
        if response.status_code == 429:
            if attempt > MAX_RETRIES:
                # Exhausted all retries — raise so the caller can surface the error.
                response.raise_for_status()

            # ── Layer 1: Parse retryDelay from the response body ─────────────
            server_delay = _parse_retry_delay(response)
            header_delay = _parse_retry_after_header(response)
            server_hint  = server_delay or header_delay

            # ── Layer 2: Exponential backoff with jitter ─────────────────────
            # Formula: min(BASE * 2^(attempt-1), MAX_WAIT) + jitter
            exp_wait = min(BASE_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
            jitter   = random.uniform(0, JITTER_S)

            # Use the LARGER of server hint and our exponential wait.
            # Never wait less than what Gemini explicitly asks for.
            wait = max(server_hint or 0.0, exp_wait) + jitter

            _log(
                f"[429] RATE LIMITED. "
                f"server_hint={server_hint}s  exp_backoff={exp_wait:.0f}s  "
                f"jitter={jitter:.1f}s  → waiting {wait:.1f}s before retry."
            )
            _log(
                f"[TIP] Free tier = 10 RPM. To avoid 429s: process fewer rows, "
                f"or upgrade to a paid tier (150 RPM)."
            )

            time.sleep(wait)
            continue

        # ── 503 / 500: Server-side transient errors ──────────────────────────
        if response.status_code in (500, 503):
            if attempt > MAX_RETRIES:
                response.raise_for_status()
            wait = min(BASE_BACKOFF_S * attempt, MAX_BACKOFF_S) + random.uniform(0, JITTER_S)
            _log(f"[{response.status_code}] Server error. Waiting {wait:.1f}s…")
            time.sleep(wait)
            continue

        # ── Non-retryable error ──────────────────────────────────────────────
        if not response.ok:
            _log(f"[ERROR] Non-retryable HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()

        # ── Success: navigate the response envelope ──────────────────────────
        # Gemini REST response shape:
        # {
        #   "candidates": [
        #     { "content": { "parts": [{"text": "<json>"}] } }
        #   ]
        # }
        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected Gemini response structure: {e}. "
                f"Body: {json.dumps(response.json(), indent=2)[:500]}"
            )

        _log(f"[HTTP] Success — received {len(text)} chars of JSON.")
        return text

    # Should never reach here but keeps type-checker happy.
    raise requests.HTTPError(f"All {MAX_RETRIES} retries exhausted.")
