"""Start a Streamlit command and wait for its health endpoint."""
from __future__ import annotations

import argparse
import socket
import subprocess
import time
import urllib.error
import urllib.request


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_healthy(port: int, proc: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"Streamlit process exited early with code {proc.returncode}.\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Streamlit server never became healthy at {health_url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", default="synth-platform-ui")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    port = _free_tcp_port()
    proc = subprocess.Popen(
        [
            args.command,
            "--server.headless=true",
            f"--server.port={port}",
            "--server.address=127.0.0.1",
            "--browser.gatherUsageStats=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_until_healthy(port, proc, args.timeout)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    print(f"Streamlit health check passed on port {port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
