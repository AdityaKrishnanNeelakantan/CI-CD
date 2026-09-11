"""OllamaChatModel: the ChatModel port backed by a local Ollama server.

Talks to a locally-running Ollama daemon (default http://localhost:11434). No
data leaves the host — this is the local-LLM deployment the architecture calls
for. Recommended local model as of 2026: qwen3:8b (Apache 2.0) or gpt-oss:20b;
the model name is configuration, not code.
"""
from __future__ import annotations

import ipaddress
import json
import urllib.request
from urllib.parse import urlparse

from synth_platform.errors import ExtractionError


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Prevent a loopback model endpoint from redirecting prompts off-host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_LOCAL_ONLY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _RejectRedirects(),
)


def validate_loopback_host(host: str) -> str:
    """Accept an HTTP Ollama endpoint only when it resolves by name to loopback."""
    parsed = urlparse(host)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama host must be a plain loopback HTTP origin.")
    hostname = parsed.hostname.casefold()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError("Remote Ollama hosts are not allowed by local-model policy.")
    return host.rstrip("/")


class OllamaChatModel:
    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.name = model
        self._url = f"{validate_loopback_host(host)}/api/chat"
        self._timeout = timeout

    def complete(self, system: str, user: str, *, json_only: bool = True) -> str:
        payload = {
            "model": self.name,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
        }
        if json_only:
            payload["format"] = "json"
        req = urllib.request.Request(
            self._url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with _LOCAL_ONLY_OPENER.open(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
        except Exception as e:
            raise ExtractionError(f"Ollama request failed: {e}") from e
        return body.get("message", {}).get("content", "")
