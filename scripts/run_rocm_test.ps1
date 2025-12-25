# Test ROCm backward pass with debugging enabled
# This script runs the minimal repro test to check if SDPA math backend fixes the hang

$projectRoot = Split-Path -Parent $PSScriptRoot
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ROCm Backward Pass Test" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Set ROCm debugging flags
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "$projectRoot"
$env:HSA_ENABLE_SDMA = "0"           # Disable SDMA
$env:HIP_LAUNCH_BLOCKING = "1"       # Get real errors instead of hangs

# Run test
Set-Location $projectRoot
& "$projectRoot\.venv_rocm\Scripts\python.exe" "$projectRoot\scripts\test_rocm_backward.py"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Test Complete" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
