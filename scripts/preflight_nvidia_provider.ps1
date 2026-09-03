param(
    [string]$Endpoint = "http://localhost:11434/v1",
    [string]$Provider = "internal",
    [string[]]$Models = @("synth-platform-slm"),
    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message =="
}

function Invoke-CurlJson {
    param(
        [string]$Uri,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [object]$Body = $null
    )

    $args = @("-sS", "--connect-timeout", "$TimeoutSeconds", "-X", $Method, $Uri)
    foreach ($key in $Headers.Keys) {
        $args += @("-H", "${key}: $($Headers[$key])")
    }
    if ($null -ne $Body) {
        $json = $Body | ConvertTo-Json -Depth 20 -Compress
        $args += @("-H", "Content-Type: application/json", "-d", $json)
    }

    $output = & curl.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed with exit code $LASTEXITCODE"
    }
    return $output
}

function Test-DirectChat {
    param([string]$Model)

    $headers = @{}
    if ($env:SP_NEMO_DATA_DESIGNER_API_KEY) {
        $headers.Authorization = "Bearer $env:SP_NEMO_DATA_DESIGNER_API_KEY"
    }
    $body = @{
        model = $Model
        messages = @(@{ role = "user"; content = "Return exactly the word ready." })
        temperature = 0
        top_p = 0.95
        max_tokens = 32
        stream = $false
    }
    try {
        $raw = Invoke-CurlJson -Uri "$Endpoint/chat/completions" -Method "POST" -Headers $headers -Body $body
        $parsed = $raw | ConvertFrom-Json
        $content = $parsed.choices[0].message.content
        [pscustomobject]@{ Layer = "direct"; Model = $Model; Ok = $true; Detail = $content }
    }
    catch {
        [pscustomobject]@{ Layer = "direct"; Model = $Model; Ok = $false; Detail = $_.Exception.Message }
    }
}

function Test-DataDesigner {
    param([string]$Model)

    $env:DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS = "1"
    $env:SP_PLATFORM_SLM_ENDPOINT = $Endpoint
    $env:SP_PLATFORM_SLM_PROVIDER = $Provider
    $env:SP_PLATFORM_SLM_MODEL = $Model
    $output = & .\.venv\Scripts\python.exe scripts\check_nvidia_data_designer.py 2>&1
    $ok = $LASTEXITCODE -eq 0
    [pscustomobject]@{
        Layer = "data-designer"
        Model = $Model
        Ok = $ok
        Detail = ($output -join "`n")
    }
}

Write-Step "Environment"
Write-Host "Endpoint: $Endpoint"
Write-Host "Provider: $Provider"
Write-Host "Provider key set: $([bool]$env:SP_NEMO_DATA_DESIGNER_API_KEY)"
Write-Host "Python: $(.\.venv\Scripts\python.exe -c 'import sys; print(sys.executable)')"

Write-Step "DNS"
Resolve-DnsName ([uri]$Endpoint).Host | Select-Object Name,Type,IPAddress | Format-Table -AutoSize

Write-Step "TCP 443"
Test-NetConnection ([uri]$Endpoint).Host -Port 443 |
    Select-Object ComputerName,RemoteAddress,RemotePort,TcpTestSucceeded |
    Format-Table -AutoSize

Write-Step "Model catalog"
$catalogRaw = Invoke-CurlJson -Uri "$Endpoint/models"
$catalog = $catalogRaw | ConvertFrom-Json
foreach ($model in $Models) {
    $listed = [bool]($catalog.data | Where-Object { $_.id -eq $model })
    [pscustomobject]@{ Model = $model; Listed = $listed } | Format-Table -AutoSize
}

if (([uri]$Endpoint).Host -notin @("localhost", "127.0.0.1", "::1") -and -not $env:SP_NEMO_DATA_DESIGNER_API_KEY) {
    Write-Warning "SP_NEMO_DATA_DESIGNER_API_KEY is not set for this non-local endpoint. Skipping authenticated direct chat and Data Designer checks."
    exit 2
}

Write-Step "Authenticated direct chat"
$directResults = foreach ($model in $Models) { Test-DirectChat -Model $model }
$directResults | Select-Object Layer,Model,Ok,Detail | Format-Table -AutoSize

Write-Step "Data Designer preview"
$ddResults = foreach ($model in $Models) { Test-DataDesigner -Model $model }
$ddResults | Select-Object Layer,Model,Ok | Format-Table -AutoSize

Write-Step "Recommendation"
$anyOk = [bool]($ddResults | Where-Object { $_.Ok })
if ($anyOk) {
    Write-Host "At least one configured model succeeded through Data Designer."
    exit 0
}
Write-Host "No configured model succeeded through Data Designer. Check endpoint, model id, provider gateway auth, proxy, firewall, or provider status."
exit 1
