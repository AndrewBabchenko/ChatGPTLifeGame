# Safe Training Runner for Life Game
# This script runs training with output logging and error handling

$ErrorActionPreference = "Continue"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Safe Training Runner" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Get script and project paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Create logs directory
$logDir = Join-Path $projectRoot "outputs\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# Generate log filename with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "training_$timestamp.log"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Output log: $logFile" -ForegroundColor Gray
Write-Host "  Models dir: outputs/checkpoints/" -ForegroundColor Gray
Write-Host "  Episodes: 50" -ForegroundColor Gray
Write-Host "  Steps/episode: 200" -ForegroundColor Gray
Write-Host ""

# Show recommendations
Write-Host "Recommendations for safe training:" -ForegroundColor Green
Write-Host "  1. Run in PowerShell (not VS Code terminal)" -ForegroundColor Gray
Write-Host "  2. Keep window open and visible" -ForegroundColor Gray
Write-Host "  3. Don't suspend/hibernate computer" -ForegroundColor Gray
Write-Host "  4. Check logs/ folder if interrupted" -ForegroundColor Gray
Write-Host "  5. Models auto-save every 10 episodes" -ForegroundColor Gray
Write-Host ""

Write-Host "`nStarting training..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop (models will be saved)" -ForegroundColor Yellow
Write-Host "Output will be shown below AND saved to $logFile`n" -ForegroundColor Gray
Write-Host "="*70

# Run training with output to both console and log file
try {
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONPATH = "$projectRoot"
    
    # ROCm debugging/stability flags
    $env:HSA_ENABLE_SDMA = "0"           # Disable SDMA for stability
    $env:HIP_LAUNCH_BLOCKING = "1"       # Get real errors instead of hangs
    # $env:PYTORCH_HIP_ALLOC_CONF = "max_split_size_mb:128"  # Optional: reduce fragmentation
    
    Set-Location $projectRoot
    & "$projectRoot\.venv_rocm\Scripts\python.exe" "$projectRoot\scripts\train.py" 2>&1 | Tee-Object -FilePath $logFile
    
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
        Write-Host "  Check outputs/checkpoints/ for saved checkpoints" -ForegroundColor Cyan
    }
} catch {
    Write-Host "`n`n========================================" -ForegroundColor Red
    Write-Host "  Training Error" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    Write-Host "  Log file: $logFile" -ForegroundColor Cyan
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
