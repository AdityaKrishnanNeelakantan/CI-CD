"""Installed Streamlit launcher for the packaged UI."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the packaged Streamlit app through ``python -m streamlit``."""
    app_path = Path(__file__).resolve().parent / "app.py"
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        *(argv if argv is not None else sys.argv[1:]),
    ]
    os.execv(sys.executable, args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
