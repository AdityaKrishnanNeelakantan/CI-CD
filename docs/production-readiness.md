# Production Readiness Runbook

This runbook is the client-facing deployment gate for Synth Platform.

The product should be launched for customers only when the readiness command
passes for the intended environment:

```powershell
$env:SP_DEPLOYMENT_PROFILE = "production"
$env:SP_PLATFORM_SLM_ENDPOINT = "https://internal-llm.example.com/v1"
$env:SP_PLATFORM_SLM_PROVIDER = "internal"
$env:SP_PLATFORM_SLM_MODEL = "synth-platform-slm"
$env:SP_PLATFORM_SLM_API_KEY_ENV = "CLIENT_LLM_API_KEY"
$env:CLIENT_LLM_API_KEY = "<set outside source control>"

.\.venv\Scripts\python.exe scripts\check_production_readiness.py
```

For an installed wheel:

```powershell
synth-platform-readiness --profile production
```

## Production Mode

Production mode is enabled with:

```text
SP_DEPLOYMENT_PROFILE=production
```

The readiness gate checks:

- mock generation is disabled;
- the OpenAI-compatible SLM endpoint is configured;
- remote SLM endpoints have an API key environment variable;
- the open-source Data Designer SDK is importable;
- the chat UI mode statuses are declared in `configs/streamlit_ui.yaml`.

Local SLM endpoints are blocked in production unless the deployment is a
controlled on-box installation:

```powershell
$env:SP_PRODUCTION_ALLOW_LOCAL_SLM = "1"
```

Use that override only when the client deployment intentionally runs Ollama,
vLLM, or another local OpenAI-compatible engine on the same host.

## Client-Facing Mode Status

Mode status is configured in `configs/streamlit_ui.yaml` under
`home.chat.modes`.

Current production posture:

| Mode | Status | Client posture |
| --- | --- | --- |
| Schema Mode | Production ready | Use for prompt/DDL to validated relational synthetic data. |
| Database Twin | Production ready | Use for source-database profiling, trained twin artifacts, distribution preservation, and FK-safe generation. |
| Customer Interactions Twin | Production ready | Use for sanitized transcript contracts and source-free synthetic interactions. |
| PDF Twin | Work in progress | Visible for roadmap/demo discussion, but disabled as a production workflow until PDF extraction/render validation is client-qualified. |

PDF mode remains marked work-in-progress because NeMo/Data Designer is not the
right core engine for PDF layout extraction/rendering. The platform can still
use Data Designer for replacement values, but the production claim should stay
limited until end-to-end PDF quality is certified with representative customer
documents.

## Launch Checklist

Before a customer demo or pilot:

1. Install from a clean environment, not the development venv.
2. Set `SP_DEPLOYMENT_PROFILE=production`.
3. Point `SP_PLATFORM_SLM_ENDPOINT` at the approved internal endpoint.
4. Set the SLM API key through an environment variable, not YAML or source code.
5. Confirm `SP_NEMO_DATA_DESIGNER_MOCK` is unset.
6. Run `synth-platform-readiness --profile production`.
7. Run the Streamlit smoke path:

```powershell
synth-platform-ui --server.port 8502
```

8. Use only modes marked `Production ready` in the chat UI for client work.

## YAML Ownership

The chat UI is intentionally YAML-driven. Customer-specific product wording,
mode availability, accepted file types, quick prompts, and customization rows
belong in `configs/streamlit_ui.yaml` or a customer-specific file referenced by:

```text
SP_STREAMLIT_UI_CONFIG=/path/to/client-ui.yaml
```

Do not hardcode customer packaging, mode status, or sales wording in Streamlit
page code.

## Go/No-Go Rule

Do not run the product with real customer data if:

- readiness status is `NOT READY`;
- mock generation is enabled;
- the UI presents a WIP mode as production-ready;
- API keys are stored in source-controlled files;
- transcript/PDF source files are committed to Git;
- validation reports are missing for generated outputs.
