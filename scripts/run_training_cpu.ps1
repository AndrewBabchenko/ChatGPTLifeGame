# Run training on CPU (stable, but slower)
# Use this if you haven't set up WSL2 yet

$projectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "  CPU Training Mode" -ForegroundColor Yellow
Write-Host "  (Stable but slower than GPU)" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

# Create log file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$projectRoot\outputs\logs\training_cpu_$timestamp.log"
New-Item -ItemType Directory -Force -Path "$projectRoot\outputs\logs" | Out-Null

Write-Host "Log file: $logFile`n" -ForegroundColor Cyan

# Run training on CPU
try {
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONPATH = "$projectRoot"
    $env:FORCE_CPU = "1"  # Force CPU mode
    
    Set-Location $projectRoot
    & "$projectRoot\.venv_rocm\Scripts\python.exe" "$projectRoot\scripts\train_advanced.py" --cpu 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n`n========================================" -ForegroundColor Green
        Write-Host "  Training Completed Successfully!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  Log file: $logFile" -ForegroundColor Cyan
        Write-Host "  Models: outputs/checkpoints/model_A_ppo.pth, model_B_ppo.pth" -ForegroundColor Cyan
    } else {
        Write-Host "`n`n========================================" -ForegroundColor Yellow
        Write-Host "  Training Ended (Exit code: $LASTEXITCODE)" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "  Log file: $logFile" -ForegroundColor Cyan
    }
} catch {
    Write-Host "`n`n========================================" -ForegroundColor Red
    Write-Host "  ERROR: Training Failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    Write-Host "  Log file: $logFile" -ForegroundColor Cyan
    exit 1
}

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
