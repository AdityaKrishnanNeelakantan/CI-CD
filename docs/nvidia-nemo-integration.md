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
pip install nvidia-nat
pip install data-designer
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
development endpoint is local and keyless:

```bash
export SP_NEMO_DATA_DESIGNER_ENDPOINT="http://localhost:8000/v1"
export SP_NEMO_DATA_DESIGNER_PROVIDER="internal"
export SP_NEMO_DATA_DESIGNER_MODEL="local/slm"
```

For customer deployments, set `SP_NEMO_DATA_DESIGNER_ENDPOINT` to the internal
NIM/LLM service. If that internal service requires authentication, set
`SP_NEMO_DATA_DESIGNER_API_KEY`. Local endpoints do not require a key.

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
| `SP_NEMO_DATA_DESIGNER_ENDPOINT` | OpenAI-compatible generation endpoint. Defaults to `http://localhost:8000/v1`. |
| `SP_NEMO_DATA_DESIGNER_PROVIDER` | Provider id used inside Data Designer. Defaults to `internal`. |
| `SP_NEMO_DATA_DESIGNER_MODEL` | Customer/internal SLM model id. Defaults to `local/slm`. |
| `SP_NEMO_DATA_DESIGNER_API_KEY` | Optional generation key when the internal endpoint requires authentication. |

## Transcript Responsibility Matrix

| Pipeline stage | Platform-owned today | Optional NVIDIA delegation |
| --- | --- | --- |
| Ingestion | Transcript parser maps text, speaker-prefixed logs, and JSON message arrays into canonical turn arrays. | None. |
| Privacy | Core contract builder redacts direct identifiers, drops raw text after contract build, and stores hashes for replay validation. | NeMo Curator/Anonymizer runs behind the adapter when installed/configured; status is stored under `privacy_policy.nvidia_nemo_curator`. |
| Identity | Transcript contract records speaker entities and source-free topic metadata. Cross-modal entity graph resolution remains platform-owned. | None. |
| Generation | Synthetic transcript generation uses contract metadata only. Relational/database volume planning remains platform-owned. | NeMo Framework/TensorRT-LLM/Triton can be plugged in later for model customization and serving. |
| Validation | Non-replay validation checks exact turn hashes, n-gram hashes, and synthetic self-similarity. | NeMo Guardrails runs behind the adapter when configured; status is stored under `nvidia_nemo_guardrails`. NeMo Evaluator remains future work. |
| Integration | Streamlit and workflow APIs depend on canonical contracts, not vendor objects. | SDK adapter layers translate vendor output back to platform models. |
