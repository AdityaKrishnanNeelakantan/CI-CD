from __future__ import annotations

import sys
import types

from synth_platform.interfaces.api.launcher import main


def test_api_launcher_invokes_uvicorn(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    fake_uvicorn = types.SimpleNamespace(run=lambda *args, **kwargs: calls.append({"args": args, "kwargs": kwargs}))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    exit_code = main(["--host", "0.0.0.0", "--port", "9000", "--reload"])

    assert exit_code == 0
    assert calls == [
        {
            "args": ("synth_platform.interfaces.api.app:app",),
            "kwargs": {"host": "0.0.0.0", "port": 9000, "reload": True},
        }
    ]
