param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$BackendDir = Join-Path $RepoRoot "backend"
$Frontend = Join-Path $RepoRoot "frontend"
$LogDir = Join-Path $RepoRoot "logs\local"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found. Run 'python -m uv sync' first."
}
if (-not (Test-Path -LiteralPath (Join-Path $Frontend "node_modules"))) {
    throw "Frontend dependencies not found. Run 'npm ci' inside frontend first."
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:SDGE_CONFIG_PATH = Join-Path $RepoRoot "config\config.yaml"
$env:SDGE_EXCEL_CONTRACTS_PATH = Join-Path $RepoRoot "config\excel_contracts.yaml"
$env:PYTHONPATH = $RepoRoot
$env:PYTHONUTF8 = "1"
$env:TORCHDYNAMO_DISABLE = "1"
$env:TORCH_COMPILE_DISABLE = "1"

$Backend = Start-Process -FilePath $Python `
    -ArgumentList @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$BackendPort") `
    -WorkingDirectory $BackendDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDir "backend.out.log") `
    -RedirectStandardError (Join-Path $LogDir "backend.err.log") `
    -PassThru

$FrontendProcess = Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort") `
    -WorkingDirectory $Frontend `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDir "frontend.out.log") `
    -RedirectStandardError (Join-Path $LogDir "frontend.err.log") `
    -PassThru

Write-Host "Backend PID: $($Backend.Id)  http://127.0.0.1:$BackendPort"
Write-Host "Frontend PID: $($FrontendProcess.Id)  http://127.0.0.1:$FrontendPort"
Write-Host "Logs: $LogDir"
