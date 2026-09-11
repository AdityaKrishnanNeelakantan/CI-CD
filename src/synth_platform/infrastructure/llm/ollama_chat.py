"""OllamaChatModel: the ChatModel port backed by a local Ollama server.

Talks to a locally-running Ollama daemon (default http://localhost:11434). No
data leaves the host — this is the local-LLM deployment the architecture calls
for.
"""
from __future__ import annotations

import json
import urllib.request

from synth_platform.errors import ExtractionError
from synth_platform.settings import Settings


class OllamaChatModel:
    def __init__(self, model: str | None = None,
                 host: str | None = None, timeout: float = 120.0):
        settings = Settings.from_env()
        self.name = model or settings.ollama_model
        self._url = f"{(host or settings.ollama_host).rstrip('/')}/api/chat"
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
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
        except Exception as e:
            raise ExtractionError(f"Ollama request failed: {e}") from e
        return body.get("message", {}).get("content", "")
