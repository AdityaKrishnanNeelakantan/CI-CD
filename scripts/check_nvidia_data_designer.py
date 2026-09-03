r"""Check OpenAI-compatible provider connectivity for direct calls and Data Designer.

Usage:
    $env:SP_PLATFORM_SLM_ENDPOINT="http://localhost:11434/v1"
    $env:SP_PLATFORM_SLM_PROVIDER="internal"
    $env:SP_PLATFORM_SLM_MODEL="synth-platform-slm"
    $env:SP_NEMO_DATA_DESIGNER_API_KEY="ollama"
    .\.venv\Scripts\python.exe scripts\check_nvidia_data_designer.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synth_platform.engine.generation.slm_runtime import resolve_platform_slm_runtime
from synth_platform.engine.generation.data_designer_provider import (
    build_data_designer_model_config,
    build_data_designer_provider,
    load_data_designer_sdk,
)


RUNTIME = resolve_platform_slm_runtime()
ENDPOINT = RUNTIME.endpoint
PROVIDER = RUNTIME.provider
MODEL = RUNTIME.model_id
API_KEY_ENV = "SP_NEMO_DATA_DESIGNER_API_KEY"


def _is_local_endpoint(endpoint: str) -> bool:
    host = (urlparse(endpoint).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def provider_api_key() -> str | None:
    key = os.getenv(API_KEY_ENV)
    if key:
        return key
    if _is_local_endpoint(ENDPOINT):
        return None
    raise RuntimeError(f"{API_KEY_ENV} is not set for non-local endpoint {ENDPOINT}.")


def check_direct_openai_compatible() -> None:
    from openai import OpenAI

    client = OpenAI(base_url=ENDPOINT, api_key=provider_api_key() or "not-needed")
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Return exactly the word ready."}],
        temperature=0,
        top_p=0.95,
        max_tokens=64,
        stream=False,
    )
    content = completion.choices[0].message.content or ""
    print(f"direct_openai_compatible=ok content={content.strip()!r}")


def check_data_designer() -> None:
    dd, DataDesigner = load_data_designer_sdk()
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            build_data_designer_model_config(
                dd,
                "provider-check",
                workflow="provider-smoke-test",
                runtime=RUNTIME,
                temperature=0,
                top_p=0.95,
            )
        ]
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="request_id",
            sampler_type=dd.SamplerType.UUID,
            params=dd.UUIDSamplerParams(short_form=True),
            convert_to="str",
        )
    )
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="connection_result",
            model_alias="provider-check",
            prompt="For request {{ request_id }}, return exactly the word ready.",
        )
    )
    provider = build_data_designer_provider(dd, RUNTIME)
    preview = DataDesigner(model_providers=[provider]).preview(builder, num_records=1)
    dataset = getattr(preview, "dataset", preview)
    frame = dataset.to_pandas() if hasattr(dataset, "to_pandas") else dataset
    print("data_designer=ok")
    print(frame)


def main() -> int:
    try:
        print(f"endpoint={ENDPOINT}")
        print(f"provider={PROVIDER}")
        print(f"model={MODEL}")
        print(f"key_required={not _is_local_endpoint(ENDPOINT)}")
        print(f"key_present={bool(os.getenv(API_KEY_ENV))}")
        check_direct_openai_compatible()
        check_data_designer()
    except Exception as exc:
        print(f"failed={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
