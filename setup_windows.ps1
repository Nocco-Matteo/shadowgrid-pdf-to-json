# setup_windows.ps1 — Setup completo da Windows.
# Avvia da PowerShell (admin):
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#
# Fa tutto: WSL2, Ubuntu, apt, venv, pip, modelli, PDF.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Shadow Grid PDF-to-JSON - Setup Windows" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. WSL2 + Ubuntu
Write-Host "[1/3] Verifica WSL2 + Ubuntu-22.04..." -ForegroundColor Yellow
$needReboot = $false
try {
    $wslList = wsl --list --verbose 2>$null
    if ($wslList -notmatch "Ubuntu-22.04") {
        Write-Host "  Ubuntu-22.04 non trovato, installo..." -ForegroundColor Yellow
        wsl --install -d Ubuntu-22.04
        $needReboot = $true
    } else {
        Write-Host "  Ubuntu-22.04 presente." -ForegroundColor Green
    }
} catch {
    Write-Host "  WSL non attivo. Abilito WSL..." -ForegroundColor Yellow
    wsl --install --no-distribution
    $needReboot = $true
}

if ($needReboot) {
    Write-Host ""
    Write-Host "  WSL2 installato. RIAVVIA WINDOWS e rilancia questo script." -ForegroundColor Red
    Write-Host "  Al riavvio ripartira da qui automaticamente." -ForegroundColor Yellow
    exit 0
}

# Verifica accesso Ubuntu
$ubuntuUser = wsl -d Ubuntu-22.04 -- whoami 2>$null
if (-not $ubuntuUser) {
    Write-Host "  ERRORE: Ubuntu-22.04 non raggiungibile. Riavvia e riprova." -ForegroundColor Red
    exit 1
}
Write-Host "  Ubuntu OK (user: $ubuntuUser)" -ForegroundColor Green

# 2. GPU
Write-Host ""
Write-Host "[2/3] Verifica GPU..." -ForegroundColor Yellow
$gpu = wsl -d Ubuntu-22.04 -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
if ($gpu) {
    Write-Host "  GPU: $gpu" -ForegroundColor Green
} else {
    Write-Host "  ATTENZIONE: GPU non visibile. Aggiorna driver NVIDIA su Windows." -ForegroundColor Red
    Write-Host "  Continuo comunque, ma vLLM non funzionera senza GPU." -ForegroundColor Yellow
}

# 3. Setup dentro Ubuntu
Write-Host ""
Write-Host "[3/3] Setup dentro Ubuntu (apt, pip, modelli ~22GB)..." -ForegroundColor Yellow
Write-Host "  Questo passo e lungo. Vai a prendere un caffe." -ForegroundColor Yellow
Write-Host ""

# Scarica lo script bash dentro Ubuntu e lancialo
$bashUrl = "https://raw.githubusercontent.com/Nocco-Matteo/shadowgrid-pdf-to-json/master/setup_inner.sh"
wsl -d Ubuntu-22.04 -- bash -c "curl -sL '$bashUrl' -o /tmp/setup_inner.sh && chmod +x /tmp/setup_inner.sh && bash /tmp/setup_inner.sh"

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
