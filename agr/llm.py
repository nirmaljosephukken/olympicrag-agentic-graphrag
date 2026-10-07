"""Gemini REST wrapper with token accounting and retry/backoff."""
from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass

import requests

from agr.config import SETTINGS

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_lock = threading.Lock()
_last = [0.0]
MIN_INTERVAL = float(__import__("os").getenv("LLM_MIN_INTERVAL", "0.0"))
# One or more keys: GEMINI_API_KEYS=key1,key2 (rotated when a key's daily quota is spent) or GEMINI_API_KEY.
_KEYS = [k.strip() for k in (__import__("os").getenv("GEMINI_API_KEYS") or SETTINGS.gemini_key).split(",") if k.strip()]
_key_idx = [0]


def _is_daily_limit(body: str) -> bool:
    """A daily quota error asks to retry in hours; a per-minute one in seconds (its text can still mention the
    daily metric, so the metric name alone is not enough)."""
    m = re.search(r'"retryDelay":\s*"(\d+)s"', body)
    if m:
        return int(m.group(1)) > 3600
    return bool(re.search(r"retry in \d+h", body))


def _rotate_key(failed: int | None = None) -> bool:
    """Move past the key at index `failed`. With parallel workers several requests can fail on the same key at once;
    only the first one advances, the others just retry on the key it moved to (otherwise they would skip good keys)."""
    with _lock:
        if failed is not None and _key_idx[0] != failed:
            return True
        if _key_idx[0] + 1 < len(_KEYS):
            _key_idx[0] += 1
            print(f"[llm] daily quota reached, switching to key #{_key_idx[0] + 1}", flush=True)
            return True
    return False


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens
    ms: float
    model: str

    def json(self) -> dict:
        d = self._json()
        if isinstance(d, list):  # some replies wrap the object in a list
            d = next((x for x in d if isinstance(x, dict)), {})
        return d if isinstance(d, dict) else {}

    def _json(self):
        t = self.text.strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", t, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {}


def generate(prompt: str, system: str | None = None, model: str | None = None, json_mode: bool = False,
             temperature: float = 0.0, max_output_tokens: int = 1024) -> LLMResult:
    model = model or SETTINGS.model
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    delay = 2.0
    for attempt in range(8):
        if MIN_INTERVAL:
            with _lock:
                wait = _last[0] + MIN_INTERVAL - time.time()
                if wait > 0:
                    time.sleep(wait)
                _last[0] = time.time()
        t0 = time.time()
        try:
            used = _key_idx[0]
            r = requests.post(URL.format(model=model), params={"key": _KEYS[used]}, json=body, timeout=120)
        except requests.RequestException:
            time.sleep(delay); delay *= 2; continue
        if r.status_code == 429 and _is_daily_limit(r.text):
            if _rotate_key(used):  # daily quota of this key is spent: move to the next configured key
                continue
            raise RuntimeError("Gemini daily quota exhausted on every configured key")
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay + random.random()); delay = min(delay * 2, 60); continue
        r.raise_for_status()
        data = r.json()
        um = data.get("usageMetadata", {})
        cand = (data.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        return LLMResult(text=text, input_tokens=um.get("promptTokenCount", 0),
                         output_tokens=um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0),
                         ms=(time.time() - t0) * 1000, model=model)
    raise RuntimeError(f"Gemini call failed after retries: {r.status_code} {r.text[:300]}")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
