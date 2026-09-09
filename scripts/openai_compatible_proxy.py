"""Log and forward OpenAI-compatible API calls to a local SLM server.

Example:
    python scripts/openai_compatible_proxy.py --target http://127.0.0.1:11434 --port 18000

Then point Data Designer at:
    SP_PLATFORM_SLM_ENDPOINT=http://127.0.0.1:18000/v1
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


class ProxyConfig:
    target: str = "http://127.0.0.1:11434"
    log_path: Path = Path("build/openai_compatible_proxy.jsonl")
    redact_authorization: bool = True


CONFIG = ProxyConfig()


class OpenAICompatibleProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _proxy(self) -> None:
        started = time.time()
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        target_url = urljoin(CONFIG.target.rstrip("/") + "/", self.path.lstrip("/"))
        request_headers = _forward_headers(self.headers)
        status_code = 502
        response_body = b""
        response_headers: dict[str, str] = {}
        error: str | None = None

        try:
            request = urllib.request.Request(
                target_url,
                data=body if self.command != "GET" else None,
                headers=request_headers,
                method=self.command,
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                status_code = int(response.status)
                response_body = response.read()
                response_headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            status_code = int(exc.code)
            response_body = exc.read()
            response_headers = dict(exc.headers.items())
            error = str(exc)
        except Exception as exc:
            response_body = json.dumps({"error": str(exc)}).encode("utf-8")
            response_headers = {"content-type": "application/json"}
            error = str(exc)

        _write_log(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method": self.command,
                "path": self.path,
                "target_url": target_url,
                "status": status_code,
                "duration_ms": round((time.time() - started) * 1000, 2),
                "request_headers": _safe_headers(dict(self.headers.items())),
                "request_json": _maybe_json(body),
                "response_headers": _safe_headers(response_headers),
                "response_json": _maybe_json(response_body),
                "response_text": None if _maybe_json(response_body) is not None else response_body.decode("utf-8", errors="replace")[:4000],
                "error": error,
            }
        )
        self._send_response(status_code, response_headers, response_body)

    def _send_response(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status_code)
        skipped = {"connection", "transfer-encoding", "content-encoding", "content-length"}
        for name, value in headers.items():
            if name.lower() not in skipped:
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _forward_headers(headers: Any) -> dict[str, str]:
    skipped = {"host", "connection", "content-length", "accept-encoding"}
    return {name: value for name, value in headers.items() if name.lower() not in skipped}


def _safe_headers(headers: dict[str, str]) -> dict[str, str]:
    safe = dict(headers)
    if CONFIG.redact_authorization:
        for name in list(safe):
            if name.lower() == "authorization":
                safe[name] = "Bearer <redacted>"
    return safe


def _maybe_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _write_log(record: dict[str, Any]) -> None:
    CONFIG.log_path.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG.log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Proxy bind host.")
    parser.add_argument("--port", type=int, default=18000, help="Proxy bind port.")
    parser.add_argument("--target", default="http://127.0.0.1:11434", help="Target OpenAI-compatible server root.")
    parser.add_argument("--log-path", default="build/openai_compatible_proxy.jsonl", help="JSONL request/response log path.")
    parser.add_argument("--show-auth", action="store_true", help="Log Authorization headers without redaction.")
    args = parser.parse_args()

    CONFIG.target = args.target
    CONFIG.log_path = Path(args.log_path)
    CONFIG.redact_authorization = not args.show_auth

    server = ThreadingHTTPServer((args.host, args.port), OpenAICompatibleProxy)
    print(f"proxy=http://{args.host}:{args.port}")
    print(f"target={CONFIG.target}")
    print(f"log_path={CONFIG.log_path}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
