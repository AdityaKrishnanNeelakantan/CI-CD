"""Installed launcher for the local FastAPI service."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Run the packaged FastAPI app through uvicorn."""
    parser = argparse.ArgumentParser(description="Run the Synth Platform FastAPI service.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for local development.")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("uvicorn is not installed. Install API dependencies with: pip install -e '.[api]'", file=sys.stderr)
        return 1

    uvicorn.run(
        "synth_platform.interfaces.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
