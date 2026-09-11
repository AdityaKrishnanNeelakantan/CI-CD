# SDK-Backed Generation Integration

This project treats open-source SDK components as optional infrastructure, not
as the platform contract. NeMo Data Designer orchestrates generation through a
configurable OpenAI-compatible model provider endpoint. In target customer
deployments, that endpoint should be the customer's internally hosted NIM/LLM
service, not a public provider endpoint.

Supported endpoint shapes include internally hosted NVIDIA NIM, vLLM, TGI,
OpenRouter-compatible gateways, Together-compatible gateways, OpenAI-compatible
gateways, or OpenAI-compatible local development servers.

All SDK output must translate back into the platform's canonical contract,
dependency profile, entity graph, artifact bundle, and validation report.

## Ownership Map

| Capability | NVIDIA package | Platform owner |
| --- | --- | --- |
| Structured/text generation | `data-designer` | generation backend adapter; output is converted into platform records |
| Document extraction and metadata | `nemo-retriever` or Docling | extraction adapter; output is converted into document templates/bindings |
| Transcript/audio curation, PII cleanup | `nemo-curator` | optional adapter in `infrastructure/integrations/nvidia_nemo.py`; transcript contracts store Curator status |
| ASR and SFT/fine-tuning | `nemo_toolkit` | future training/text adapters |
| Retrieval/tool planning | `nvidia-nat` | future application workflow adapter |
| SSOT grounding and refusal policy | `nemoguardrails` | optional transcript validation adapter in `infrastructure/integrations/nvidia_nemo.py` |

## Install Notes

Keep these as optional extras or environment-specific installs. Do not import
them from core modules or Streamlit page load paths.

```bash
pip install data-designer
```

For native Windows local SLM testing, install only `data-designer` and point it
at Ollama or vLLM through the OpenAI-compatible endpoint. Optional NVIDIA
curation/guardrails packages can pull dependencies such as `uvloop`, which do
not support native Windows.

Install the broader optional NVIDIA stack only in WSL2/Linux, an NVIDIA PyTorch
container, or another package-supported environment:

```bash
pip install nvidia-nat
pip install nemoguardrails
uv pip install 'nemo-curator[text_cpu]'
uv pip install 'nemo-curator[audio_cpu]'
pip install 'nemo_toolkit[asr]'
```

The project also exposes a conservative optional extra:

```bash
pip install .[nvidia]
```

NeMo Curator should be installed by modality. Avoid one all-extras install,
because its optional dependency sets can conflict. For production NeMo Toolkit
workloads, prefer an NVIDIA PyTorch container or a dedicated Python environment
with the supported Python version for the target package.

## Runtime Rule

Data Designer must be configured with a model provider endpoint. The default
development endpoint is local and keyless. One shared SLM runtime is used by
Schema Mode, PDF Twin, and Customer Interactions Twin:

```bash
export SP_PLATFORM_SLM_ENDPOINT="http://localhost:11434/v1"
export SP_PLATFORM_SLM_PROVIDER="internal"
export SP_PLATFORM_SLM_MODEL="synth-platform-slm"
```

For customer deployments, set `SP_PLATFORM_SLM_ENDPOINT` to the internal
NIM/LLM service. If that internal service requires authentication, set
`SP_PLATFORM_SLM_API_KEY_ENV` to the environment-variable name holding the key.
Local endpoints do not require a key. The older `SP_NEMO_DATA_DESIGNER_*`
environment variables remain supported only as compatibility fallbacks.

Every SDK result must be adapted back into platform-owned contracts:

```text
SDK output
  -> CanonicalContract
  -> DependencyProfile
  -> EntityGraph
  -> SynthArtifact
  -> ValidationReport
```

Provider metadata must never become the durable platform contract.

## Transcript Runtime Configuration

The Customer Interactions Twin UI can use SDK-backed generation when the packages
are installed and configured. If an optional package is missing or not
configured, the adapter records the status in the contract or validation report
and the platform continues with its built-in safeguards where that fallback is
allowed.

| Setting | Purpose |
| --- | --- |
| `SP_NVIDIA_NEMO_ENABLED` | Enables or disables optional NVIDIA SDK use in the UI. |
| `SP_NVIDIA_CURATOR_BASE_URL` | Uses NeMo Curator `LLMPiiModifier` against a configured NIM/OpenAI-compatible endpoint. |
| `SP_NVIDIA_CURATOR_API_KEY` | Optional API key for the Curator endpoint. |
| `SP_NVIDIA_CURATOR_MODEL` | Curator LLM model name; defaults to `meta/llama-3.1-70b-instruct`. |
| `SP_NVIDIA_GUARDRAILS_CONFIG_PATH` | NeMo Guardrails config directory or file for transcript validation. |
| `SP_PLATFORM_SLM_ENDPOINT` | Shared OpenAI-compatible SLM endpoint for all Data Designer workflows. Defaults to `http://localhost:11434/v1`. |
| `SP_PLATFORM_SLM_PROVIDER` | Shared provider id used inside Data Designer. Defaults to `internal`. |
| `SP_PLATFORM_SLM_MODEL` | Shared customer/internal SLM model id. Defaults to `synth-platform-slm`. |
| `SP_PLATFORM_SLM_API_KEY_ENV` | Optional name of the environment variable holding the shared SLM key. Defaults to `SP_NEMO_DATA_DESIGNER_API_KEY`. |
| `SP_SCHEMA_ROW_GENERATION_MODE` | Schema row-generation engine. Defaults to `deterministic`; set `llm` only for Data Designer row experiments. |
| `SP_SCHEMA_DRAFT_CACHE` | Enables prompt+model schema draft cache under `build/schema_draft_cache`. Defaults to `1`. |
| `SP_SCHEMA_DRAFT_MAX_TOKENS` | Schema draft token cap. Defaults to `768`. |
| `SP_TRANSCRIPT_DATA_DESIGNER_MAX_TOKENS` | Per-turn transcript message token cap. Defaults to `96` for local SLM latency. |
| `SP_TRANSCRIPT_DATA_DESIGNER_TIMEOUT` | Data Designer per-request timeout in seconds. Defaults to `180`. |
| `SP_TRANSCRIPT_DATA_DESIGNER_MAX_PARALLEL_REQUESTS` | Data Designer transcript request parallelism. Defaults to `4`. |
| `SP_TRANSCRIPT_DATA_DESIGNER_TEMPERATURE` | Transcript generation temperature. Defaults to `0.2`. |
| `SP_TRANSCRIPT_DATA_DESIGNER_TOP_P` | Transcript generation top-p. Defaults to `0.9`. |
| `SP_PDF_DATA_DESIGNER_BINDINGS_PER_BATCH` | Number of PDF binding seed rows sent to one Data Designer preview call. Defaults to `10` for local SLM latency. |
| `SP_PDF_DATA_DESIGNER_MAX_VALUE_CHARS` | Maximum length accepted by the PDF value validation column. Defaults to `240`. |
| `SP_TRANSCRIPT_SSOT_MAX_TOKENS` | Structured SSOT extraction token cap. Defaults to `768`. |

## Transcript Responsibility Matrix

| Pipeline stage | Platform-owned today | Optional NVIDIA delegation |
| --- | --- | --- |
| Ingestion | Transcript parser maps text, speaker-prefixed logs, and JSON message arrays into canonical turn arrays. | None. |
| Privacy | Core contract builder redacts direct identifiers, drops raw text after contract build, and stores hashes for replay validation. | NeMo Curator/Anonymizer runs behind the adapter when installed/configured; status is stored under `privacy_policy.nvidia_nemo_curator`. |
| Identity | Transcript contract records speaker entities and source-free topic metadata. Cross-modal entity graph resolution remains platform-owned. | NeMo Data Designer can enrich the contract with a fixed structured SSOT object using `LLMStructuredColumnConfig` over sanitized turn text. |
| Generation | Synthetic transcript generation uses contract metadata only. Relational/database volume planning remains platform-owned. | NeMo Data Designer uses a `DataFrameSeedSource`, one `LLMTextColumnConfig` per seeded turn, and a `ValidationColumnConfig` before results return to the platform. |
| Validation | Non-replay validation checks exact turn hashes, n-gram hashes, and synthetic self-similarity. | NeMo Guardrails runs behind the adapter when configured; status is stored under `nvidia_nemo_guardrails`. NeMo Evaluator remains future work. |
| Integration | Streamlit and workflow APIs depend on canonical contracts, not vendor objects. | SDK adapter layers translate vendor output back to platform models. |

## PDF Runtime Pattern

PDF extraction uses the selected document extraction SDK path before the platform
builds a redacted binding map. Synthetic PDF value generation then uses Data
Designer directly with seed rows:

```text
binding map row -> DataFrameSeedSource -> LLMTextColumnConfig(synthetic_value)
                -> ValidationColumnConfig -> document_synthetic_values.json
```

Each seed row represents one field, table cell, or inline span binding. Data
Designer generates only the replacement value for that row, so the platform no
longer asks the model to reproduce nested PDF/table JSON structures.

## Local Payload Logging

Use `scripts/openai_compatible_proxy.py` to inspect raw OpenAI-compatible
request/response payloads while keeping the model server local.

For Ollama:

```powershell
ollama serve
.\.venv\Scripts\python.exe scripts\openai_compatible_proxy.py --target http://127.0.0.1:11434 --port 18000
$env:SP_PLATFORM_SLM_ENDPOINT="http://127.0.0.1:18000/v1"
$env:SP_PLATFORM_SLM_MODEL="synth-platform-slm"
$env:SP_NEMO_DATA_DESIGNER_API_KEY="ollama"
```

For vLLM:

```powershell
vllm serve meta-llama/Llama-3.1-8B-Instruct --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe scripts\openai_compatible_proxy.py --target http://127.0.0.1:8000 --port 18000
$env:SP_PLATFORM_SLM_ENDPOINT="http://127.0.0.1:18000/v1"
$env:SP_PLATFORM_SLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
```

Proxy logs are written to `build/openai_compatible_proxy.jsonl`.
