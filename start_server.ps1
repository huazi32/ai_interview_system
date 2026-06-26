$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PATH = (Join-Path $PSScriptRoot ".runtime\Scripts") + [IO.Path]::PathSeparator + $env:PATH

Write-Host "Starting AI interview system..."
Write-Host "Home: http://127.0.0.1:28080/"
Write-Host "Live: http://127.0.0.1:28080/live"
Write-Host ""

& (Join-Path $PSScriptRoot ".runtime\Scripts\python.exe") "main.py"
