param(
  [string]$GenerationKey = "",
  [string]$Endpoint = "http://localhost:11434/v1",
  [string]$Provider = "internal",
  [string]$Model = "synth-platform-slm",
  [switch]$InstallCuration
)

$ErrorActionPreference = "Stop"

if ($GenerationKey) {
  $env:SP_NEMO_DATA_DESIGNER_API_KEY = $GenerationKey
}

$env:SP_PLATFORM_SLM_ENDPOINT = $Endpoint
$env:SP_PLATFORM_SLM_PROVIDER = $Provider
$env:SP_PLATFORM_SLM_MODEL = $Model
$env:SP_NVIDIA_GUARDRAILS_CONFIG_PATH = "configs\nemo_guardrails"

$pythonInfoScript = @'
import platform
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
print(platform.system())
'@
$pythonInfo = $pythonInfoScript | python -
$pythonVersion = [version]$pythonInfo[0]
$platformName = $pythonInfo[1]

python -m pip install "data-designer==0.9.1"

if ($InstallCuration) {
  if ($platformName -eq "Windows") {
    Write-Warning "Skipping nemoguardrails/nemo-curator on native Windows because transitive dependencies such as uvloop do not support Windows. Use WSL2/Linux or an NVIDIA container for curation/guardrails."
  } elseif ($pythonVersion -ge [version]"3.14") {
    Write-Warning "Skipping nemoguardrails/nemo-curator on Python $pythonVersion. Use Python 3.12/3.13 for optional NVIDIA curation/guardrails SDKs."
  } else {
    python -m pip install "nemoguardrails"
    python -m pip install "nemo-curator[text_cpu]"
  }
} else {
  Write-Host "Skipping optional curation/guardrails installs. Pass -InstallCuration on Linux/WSL with a supported Python version."
}

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
Write-Host "  SP_PLATFORM_SLM_ENDPOINT=$env:SP_PLATFORM_SLM_ENDPOINT"
Write-Host "  SP_PLATFORM_SLM_PROVIDER=$env:SP_PLATFORM_SLM_PROVIDER"
Write-Host "  SP_PLATFORM_SLM_MODEL=$env:SP_PLATFORM_SLM_MODEL"
Write-Host "  SP_NVIDIA_GUARDRAILS_CONFIG_PATH=$env:SP_NVIDIA_GUARDRAILS_CONFIG_PATH"
