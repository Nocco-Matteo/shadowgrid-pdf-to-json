# setup_windows.ps1 - Setup completo da Windows.
# Avvia da PowerShell (admin):
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -SkipSmoke
#
# Fa tutto: WSL2, Ubuntu, apt, venv, pip, modelli, PDF, smoke test reale.

param([switch]$SkipSmoke)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Shadow Grid PDF-to-JSON - Setup Windows" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. WSL2 + Ubuntu
Write-Host "[1/4] Verifica WSL2 + Ubuntu-22.04..." -ForegroundColor Yellow
$needReboot = $false

# wsl --list ritorna UTF-16; prova direttamente a lanciare un comando
$ubuntuUser = wsl -d Ubuntu-22.04 -- whoami 2>$null
if (-not $ubuntuUser) {
    Write-Host "  Ubuntu-22.04 non trovato, installo..." -ForegroundColor Yellow
    wsl --install -d Ubuntu-22.04
    $needReboot = $true
} else {
    Write-Host "  Ubuntu-22.04 presente (user: $ubuntuUser)" -ForegroundColor Green
}

if ($needReboot) {
    Write-Host ""
    Write-Host "  WSL2 installato. RIAVVIA WINDOWS e rilancia questo script." -ForegroundColor Red
    Write-Host "  Al riavvio ripartira da qui automaticamente." -ForegroundColor Yellow
    exit 0
}

# 2. GPU
Write-Host ""
Write-Host "[2/4] Verifica GPU..." -ForegroundColor Yellow
$gpu = wsl -d Ubuntu-22.04 -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
if ($gpu) {
    Write-Host "  GPU: $gpu" -ForegroundColor Green
} else {
    Write-Host "  ATTENZIONE: GPU non visibile. Aggiorna driver NVIDIA su Windows." -ForegroundColor Red
    Write-Host "  Continuo comunque, ma vLLM non funzionera senza GPU." -ForegroundColor Yellow
}

# 3. Setup dentro Ubuntu
Write-Host ""
Write-Host "[3/4] Setup dentro Ubuntu (apt, pip, modelli ~22GB)..." -ForegroundColor Yellow
Write-Host "  Questo passo e lungo. Vai a prendere un caffe." -ForegroundColor Yellow
Write-Host ""

# Usa i file locali della repo (quella da cui lanci questo script):
# passa setup_inner.sh a bash via stdin e il path del repo come argomento.
# Niente download da GitHub -> setup identico al codice che hai in locale.
$wslPath = (wsl -d Ubuntu-22.04 -- wslpath -a ($PSScriptRoot -replace '\\','/')).Trim()
Write-Host "  Repo: $PSScriptRoot -> $wslPath" -ForegroundColor Cyan
$script = (Get-Content "$PSScriptRoot\setup_inner.sh" -Raw) -replace "`r`n", "`n"
$script | wsl -d Ubuntu-22.04 -- bash -s -- "$wslPath"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Setup interno FALLITO (exit code $LASTEXITCODE) - vedi l'output sopra." -ForegroundColor Red
    Write-Host "  Il setup non e' completo: correggi e rilancia." -ForegroundColor Red
    exit 1
}
Write-Host "  Setup interno OK." -ForegroundColor Green

# 4. Smoke test reale (modelli veri, ~15-30 min: ogni modello va caricato in VRAM)
if ($SkipSmoke) {
    Write-Host ""
    Write-Host "[4/4] Smoke test saltato (-SkipSmoke)." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "[4/4] Smoke test reale (vLLM + modelli veri, ~15-30 min)..." -ForegroundColor Yellow
    wsl -d Ubuntu-22.04 -- bash -c "cd ~/shadowgrid-pdf-to-json && source .venv/bin/activate && python scripts/smoke_real.py"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Smoke test OK" -ForegroundColor Green
    } else {
        Write-Host "  Smoke test FALLITO - vedi sopra e runs/vllm_*.log" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETATO" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Per usare la pipeline:" -ForegroundColor Cyan
Write-Host "  wsl -d Ubuntu-22.04" -ForegroundColor White
Write-Host "  cd ~/shadowgrid-pdf-to-json" -ForegroundColor White
Write-Host "  source .venv/bin/activate" -ForegroundColor White
Write-Host "  python -m pipeline.cli run data/pdfs/NOME_FILE.pdf" -ForegroundColor White
Write-Host ""
Write-Host "Smoke test (vero, GPU) rilanciabile con:" -ForegroundColor Cyan
Write-Host "  python scripts/smoke_real.py [--skip-ocr-b] [--pdf file.pdf]" -ForegroundColor White
Write-Host ""
