"""Minimal TigerGraph 4.x / Savanna REST client (requests only)."""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests

from agr.config import SETTINGS, ROOT


class TigerGraphError(RuntimeError):
    pass


_local = threading.local()


def take_transports() -> list[str]:
    """Which transports served the queries made on this thread since the last call (for the trace)."""
    v = getattr(_local, "vias", [])
    _local.vias = []
    return v


class TG:
    def __init__(self, s=SETTINGS):
        if not s.tg_host:
            raise TigerGraphError("TG_HOST is not set (copy .env.example to .env)")
        self.s, self.base, self.graph = s, s.tg_host, s.tg_graph
        self.http = requests.Session()
        self._token = s.tg_token or self._cached_token()
        self.calls = 0

    # -- auth -------------------------------------------------------------
    def _cached_token(self) -> str | None:
        p = ROOT / ".tg_token"
        if p.exists() and time.time() - p.stat().st_mtime < 20 * 86400:
            return p.read_text().strip() or None
        return None

    def wait_until_awake(self, max_wait: float = 300):
        t0 = time.time()
        while True:
            try:
                r = self.http.get(f"{self.base}/api/ping", timeout=30)
                ok = r.status_code == 200 and "pong" in r.text
            except requests.RequestException:
                ok = False
            if ok:
                return
            if time.time() - t0 > max_wait:
                raise TigerGraphError("Savanna workspace still starting; resume it in the Savanna console")
            print("[tigergraph] workspace starting, waiting 10s ...", flush=True)
            time.sleep(10)

    def token(self) -> str:
        if self._token:
            return self._token
        self.wait_until_awake()
        t0 = time.time()
        while True:
            r = self.http.post(f"{self.base}/gsql/v1/tokens",
                               json={"secret": self.s.tg_secret, "lifetime": "2592000"}, timeout=60)
            if r.ok and "token" in r.text:
                break
            if time.time() - t0 > 300:
                raise TigerGraphError(f"token request failed: {r.status_code} {r.text[:200]}")
            time.sleep(10)
        self._token = r.json()["token"]
        (ROOT / ".tg_token").write_text(self._token)
        return self._token

    def _h(self, extra: dict | None = None) -> dict:
        h = {"Authorization": f"Bearer {self.token()}", "GSQL-TIMEOUT": "600000"}
        h.update(extra or {})
        return h

    def _req(self, method: str, url: str, **kw) -> requests.Response:
        for attempt in range(6):
            r = self.http.request(method, url, **kw)
            if r.status_code in (502, 503, 504) or "Starting workspace" in r.text[:300]:
                self.wait_until_awake()
                continue
            if r.status_code == 401 and attempt == 0:
                self._token = None
                kw["headers"]["Authorization"] = f"Bearer {self.token()}"
                continue
            return r
        return r

    # -- api --------------------------------------------------------------
    def gsql(self, statements: str, timeout: int = 1800) -> str:
        r = self._req("POST", f"{self.base}/gsql/v1/statements", data=statements.encode("utf-8"),
                      headers=self._h({"Content-Type": "text/plain"}), timeout=timeout)
        if r.status_code >= 400:
            raise TigerGraphError(f"GSQL HTTP {r.status_code}: {r.text[:1000]}")
        return r.text

    def query(self, name: str, params: dict[str, Any] | None = None, timeout: int = 120) -> list[dict]:
        """Run an installed GSQL query. TG_TRANSPORT=mcp routes it through the TigerGraph MCP server
        (agr.mcp_bridge); on any MCP failure, or with the default TG_TRANSPORT=rest, RESTPP is used."""
        params = params or {}
        if os.getenv("TG_TRANSPORT", "rest").lower() == "mcp" and self._mcp_ready():
            try:
                with self._mcp_lock:
                    out = self._mcp.run_installed_query(name, params)
                self.calls += 1
                _local.vias = getattr(_local, "vias", []) + ["mcp"]
                return out
            except Exception as e:  # fall back to RESTPP, and say so in the trace
                _local.vias = getattr(_local, "vias", []) + [f"rest(fallback: {type(e).__name__})"]
        else:
            _local.vias = getattr(_local, "vias", []) + ["rest"]
        return self._rest_query(name, params, timeout)

    _mcp = None
    _mcp_failed = False
    _mcp_lock = threading.Lock()
    _mcp_init_lock = threading.Lock()

    def _mcp_ready(self) -> bool:
        if self._mcp is None and not self._mcp_failed:
            with TG._mcp_init_lock:  # one MCP server per process, even with several worker threads
                if self._mcp is None and not self._mcp_failed:
                    self._start_mcp()
        return self._mcp is not None

    def _start_mcp(self):
        if True:
            try:
                from agr.mcp_bridge import TigerGraphMCP
                TG._mcp = TigerGraphMCP(token=self.token())
            except Exception as e:
                TG._mcp_failed = True
                print(f"[tigergraph] MCP unavailable, using RESTPP: {e}", flush=True)

    def _rest_query(self, name: str, params: dict[str, Any], timeout: int = 120) -> list[dict]:
        url = f"{self.base}/restpp/query/{self.graph}/{name}"
        self.calls += 1
        if any(isinstance(v, (list, tuple)) for v in params.values()):
            r = self._req("POST", url, data=json.dumps(params),
                          headers=self._h({"Content-Type": "application/json"}), timeout=timeout)
        else:
            q = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in params.items()}
            qs = urllib.parse.urlencode(q, quote_via=urllib.parse.quote)
            r = self._req("GET", f"{url}?{qs}" if qs else url, headers=self._h(), timeout=timeout)
        if r.status_code >= 400:
            raise TigerGraphError(f"query {name} HTTP {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("error"):
            raise TigerGraphError(f"query {name}: {body.get('message')}")
        return body.get("results", [])

    def upsert(self, vertices: dict | None = None, edges: dict | None = None) -> dict:
        payload = {k: v for k, v in (("vertices", vertices), ("edges", edges)) if v}
        r = self._req("POST", f"{self.base}/restpp/graph/{self.graph}", data=json.dumps(payload),
                      headers=self._h({"Content-Type": "application/json"}), timeout=600)
        if r.status_code >= 400:
            raise TigerGraphError(f"upsert HTTP {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("error"):
            raise TigerGraphError(f"upsert: {body.get('message')}")
        return body.get("results", [{}])[0]

    def count(self, vtype: str) -> int:
        r = self._req("GET", f"{self.base}/restpp/graph/{self.graph}/vertices/{vtype}?count_only=true",
                      headers=self._h(), timeout=60)
        return int(r.json()["results"][0]["count"])
