# Manual SDK-Backed Mode Benchmark

Use `scripts/benchmark_nemo_modes.py` to manually exercise and compare the four
platform modes: Schema, Database, PDF, and Customer Interactions/Transcript.

From this repo:

```bash
python scripts/benchmark_nemo_modes.py --rows 60 --model-type safe_gaussian_copula
```

The script writes:

```text
build/manual_nemo_benchmark/benchmark_report.json
```

To run generation against a customer/internal OpenAI-compatible endpoint:

```bash
export SP_NEMO_DATA_DESIGNER_ENDPOINT="http://localhost:8000/v1"
export SP_NEMO_DATA_DESIGNER_PROVIDER="internal"
export SP_NEMO_DATA_DESIGNER_MODEL="local/slm"
python scripts/benchmark_nemo_modes.py --rows 60 --model-type safe_gaussian_copula
```

To enable optional curation/guardrails hooks used by Customer Interactions Twin:

```bash
python scripts/benchmark_nemo_modes.py \
  --nvidia-enabled \
  --curator-base-url http://localhost:8000/v1 \
  --curator-model meta/llama-3.1-70b-instruct \
  --guardrails-config-path path/to/guardrails/config
```

Notes:

- The current optional SDK extra is guarded for Python versions below 3.14.
- NeMo Curator and NeMo Guardrails are wired into the Customer Interactions mode.
- Schema, Database, and PDF modes are still measured in the same report so you
  can compare end-to-end local workflow performance across all four modes.
- Use `--only schema database` to run a subset while iterating.
