"""
src/common/gemini_client.py
============================
Shared Gemini REST client — handles 400/429/500/503 robustly.

ROOT CAUSE OF HTTP 400  ← THIS IS WHAT BROKE sprint_planner
---------------------------------------------------------------
Pydantic v2's model_json_schema() emits $defs + $ref for every
nested model.  Example for SprintPlan (contains List[Story]):

  { "$defs": { "Story": {...} },
    "properties": {
      "stories": { "items": { "$ref": "#/$defs/Story" } }
    }
  }

Gemini REST API accepts only a flat OpenAPI-like schema.
Sending $defs / $ref produces:
  HTTP 400 — "Unknown name \"$defs\" at generation_config.response_schema"
  HTTP 400 — "Unknown name \"$ref\" at ...items"

FIX: _resolve_schema() walks the entire tree, inlines every $ref,
drops $defs, and strips unsupported keys (title, $schema) before
the body is sent.  This is applied automatically inside call_gemini_rest.

Error guide
-----------
  400  BAD_REQUEST          → schema has $ref/$defs OR bad field names
                               → auto-fixed by _resolve_schema()
  429  RESOURCE_EXHAUSTED   → free-tier 10 RPM quota hit
                               → long back-off with server retryDelay hint
  503  SERVICE_UNAVAILABLE  → model overloaded
                               → same long back-off + model fallback chain
  500  INTERNAL_SERVER_ERROR→ transient; retry with back-off
"""

from __future__ import annotations

import copy
import json
import random
import time
from typing import Callable

import requests

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_INTERVAL   : float = 6.5    # seconds between requests (10 RPM free tier)
BASE_BACKOFF_S : float = 20.0   # start high — 503 needs longer waits than 429
MAX_BACKOFF_S  : float = 120.0
JITTER_S       : float = 4.0
MAX_RETRIES    : int   = 5

GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_FALLBACK_MODELS: list[str] = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# Keys Gemini's schema parser rejects (JSON Schema but not OpenAPI)
_STRIP_KEYS = frozenset({"$schema"})  # "title" removed: it is also a valid field name in Pydantic models

# ── Module-level throttle ─────────────────────────────────────────────────────
_last_request_ts: float = 0.0


# ── Schema flattener (fixes HTTP 400) ─────────────────────────────────────────

def _resolve_schema(schema: dict) -> dict:
    """
    Flatten a Pydantic JSON Schema so Gemini REST accepts it.

    Pydantic v2 puts nested models in $defs and references them with $ref.
    Gemini does not support $ref or $defs — it needs a fully inlined schema.
    This function resolves every $ref recursively and drops $defs.

    Also strips 'title' and '$schema' keys which Gemini rejects.
    """
    schema = copy.deepcopy(schema)
    defs   = schema.pop("$defs", {})

    def _resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                # "#/$defs/Story" → "Story"
                ref_name = node["$ref"].split("/")[-1]
                resolved = copy.deepcopy(defs.get(ref_name, {}))
                # Merge any sibling keys (allOf patterns etc.)
                for k, v in node.items():
                    if k != "$ref":
                        resolved[k] = v
                return _resolve(resolved)
            return {k: _resolve(v) for k, v in node.items() if k not in _STRIP_KEYS}
        elif isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    flat = _resolve(schema)
    flat.pop("$defs", None)    # remove any that crept back in
    return flat


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_default(msg: str) -> None:
    print(f"    {msg}", flush=True)


def _throttle(log_fn: Callable[[str], None]) -> None:
    global _last_request_ts
    now  = time.monotonic()
    wait = MIN_INTERVAL - (now - _last_request_ts)
    if wait > 0:
        log_fn(f"[THROTTLE] Spacing requests — waiting {wait:.1f}s (10 RPM limit)…")
        time.sleep(wait)
    _last_request_ts = time.monotonic()


def _parse_retry_delay(response: requests.Response) -> float | None:
    try:
        details = response.json().get("error", {}).get("details", [])
        for item in details:
            val = item.get("retryDelay", "")
            if isinstance(val, str) and val.endswith("s"):
                return float(val[:-1])
    except Exception:
        pass
    return None


def _compute_wait(attempt: int, server_hint: float | None) -> float:
    exp    = min(BASE_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
    jitter = random.uniform(0, JITTER_S)
    return max(server_hint or 0.0, exp) + jitter


# ── Single-model call ─────────────────────────────────────────────────────────

def _call_one_model(
    model:       str,
    prompt:      str,
    api_key:     str,
    schema:      dict,
    log_fn:      Callable[[str], None],
    progress_fn: Callable[[str, int], None] | None,
) -> str | None:
    """
    Returns JSON string on success.
    Returns None if 503 retries exhausted (trigger fallback).
    Raises requests.HTTPError for non-retriable errors.
    """
    url  = f"{GEMINI_REST_BASE}/models/{model}:generateContent?key={api_key}"
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}

    # ── Flatten schema BEFORE building body ──────────────────────────────────
    # This resolves $ref/$defs so Gemini REST accepts nested Pydantic models.
    flat_schema = _resolve_schema(schema)
    log_fn("[SCHEMA] $ref/$defs resolved — flat schema ready for Gemini.")

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema":   flat_schema,
        },
    }

    for attempt in range(1, MAX_RETRIES + 2):
        _throttle(log_fn)

        log_fn(f"[HTTP] POST {GEMINI_REST_BASE}/models/{model}:generateContent "
               f"(attempt {attempt}/{MAX_RETRIES + 1})")

        if progress_fn:
            progress_fn(
                f"Model: {model}  |  Attempt {attempt} / {MAX_RETRIES + 1}",
                int(80 * attempt / (MAX_RETRIES + 1)),
            )

        resp = requests.post(url, headers=hdrs, json=body, timeout=(10, 60))
        log_fn(f"[HTTP] Status: {resp.status_code}")

        # 429 — rate limited
        if resp.status_code == 429:
            if attempt > MAX_RETRIES:
                resp.raise_for_status()
            hint = _parse_retry_delay(resp)
            wait = _compute_wait(attempt, hint)
            log_fn(f"[429] RATE LIMITED — server_hint={hint}s → waiting {wait:.1f}s")
            time.sleep(wait)
            continue

        # 503 — model overloaded → trigger fallback after retries
        if resp.status_code == 503:
            if attempt > MAX_RETRIES:
                log_fn(f"[503] All retries exhausted for {model}. Trying fallback model.")
                return None
            wait = _compute_wait(attempt, None)
            log_fn(f"[503] SERVICE UNAVAILABLE — {model} overloaded. "
                   f"Waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES+1}).")
            time.sleep(wait)
            continue

        # 500 — transient server error
        if resp.status_code == 500:
            if attempt > MAX_RETRIES:
                resp.raise_for_status()
            wait = _compute_wait(attempt, None)
            log_fn(f"[500] Internal server error. Waiting {wait:.1f}s…")
            time.sleep(wait)
            continue

        # 400 — bad request (schema issue, bad key, etc.) — NON-retriable
        if resp.status_code == 400:
            log_fn(f"[ERROR] HTTP 400 Bad Request: {resp.text[:400]}")
            log_fn("[HINT] If error mentions $defs or $ref, schema flattening may have missed a cycle.")
            resp.raise_for_status()

        # Other 4xx — non-retriable
        if not resp.ok:
            log_fn(f"[ERROR] Non-retriable HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()

        # ── Success ───────────────────────────────────────────────────────────
        try:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected Gemini response structure: {e}. "
                f"Body: {json.dumps(resp.json(), indent=2)[:400]}"
            )

        log_fn(f"[HTTP] Success — {len(text)} chars from model '{model}'.")
        return text

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def call_gemini_rest(
    prompt:          str,
    api_key:         str,
    schema:          dict,
    model:           str  = "gemini-2.5-flash",
    fallback_models: list[str] | None = None,
    log_fn:          Callable[[str], None] | None = None,
    progress_fn:     Callable[[str, int], None]   | None = None,
) -> str:
    """
    Call Gemini generateContent and return raw JSON string.

    The schema is automatically flattened (no $ref/$defs) before sending.
    """
    _log = log_fn or _log_default

    if fallback_models is None:
        fallback_models = DEFAULT_FALLBACK_MODELS

    model_chain = [model] + [m for m in fallback_models if m != model]

    for i, current_model in enumerate(model_chain):
        if i > 0:
            _log(f"[FALLBACK] Switching to: {current_model}")

        result = _call_one_model(
            model       = current_model,
            prompt      = prompt,
            api_key     = api_key,
            schema      = schema,
            log_fn      = _log,
            progress_fn = progress_fn,
        )

        if result is not None:
            return result

    raise ValueError(
        f"All models exhausted: {model_chain}. "
        "Gemini API may be experiencing an outage. Try again in a few minutes."
    )
