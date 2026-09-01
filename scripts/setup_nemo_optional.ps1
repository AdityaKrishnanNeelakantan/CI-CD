param(
  [string]$GenerationKey = "",
  [string]$Endpoint = "http://localhost:8000/v1",
  [string]$Provider = "internal",
  [string]$Model = "local/slm"
)

$ErrorActionPreference = "Stop"

if ($GenerationKey) {
  $env:SP_NEMO_DATA_DESIGNER_API_KEY = $GenerationKey
}

$env:SP_NEMO_DATA_DESIGNER_ENDPOINT = $Endpoint
$env:SP_NEMO_DATA_DESIGNER_PROVIDER = $Provider
$env:SP_NEMO_DATA_DESIGNER_MODEL = $Model
$env:SP_NVIDIA_GUARDRAILS_CONFIG_PATH = "configs\nemo_guardrails"

python -m pip install "data-designer==0.9.1"
python -m pip install "nemoguardrails"
python -m pip install "nemo-curator[text_cpu]"

$checkScript = @'
import importlib.util

packages = {
    "data_designer": "Data Designer",
    "nemoguardrails": "NeMo Guardrails",
    "nemo_curator": "NeMo Curator",
}

for import_name, label in packages.items():
    print(f"{label}: {'installed' if importlib.util.find_spec(import_name) else 'missing'}")
'@

$checkScript | python -

Write-Host ""
Write-Host "Configured:"
Write-Host "  SP_NEMO_DATA_DESIGNER_ENDPOINT=$env:SP_NEMO_DATA_DESIGNER_ENDPOINT"
Write-Host "  SP_NEMO_DATA_DESIGNER_PROVIDER=$env:SP_NEMO_DATA_DESIGNER_PROVIDER"
Write-Host "  SP_NEMO_DATA_DESIGNER_MODEL=$env:SP_NEMO_DATA_DESIGNER_MODEL"
Write-Host "  SP_NVIDIA_GUARDRAILS_CONFIG_PATH=$env:SP_NVIDIA_GUARDRAILS_CONFIG_PATH"
