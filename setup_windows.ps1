# setup_windows.ps1 — Setup completo della pipeline da Windows.
# Salva questo file sul desktop e lancialo da PowerShell come amministratore:
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#
# Fa tutto:
#   1. Verifica/abilita WSL2
#   2. Installa Ubuntu-22.04 se manca
#   3. Dentro Ubuntu: apt, venv, pip, download modelli, copia PDF
#   4. Alla fine sei pronto per girare la pipeline

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Shadow Grid PDF-to-JSON — Setup Windows" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Verifica WSL2 ---
Write-Host "[1/4] Verifica WSL2..." -ForegroundColor Yellow
$wslInstalled = $false
try {
    $wslList = wsl --list --verbose 2>$null
    if ($wslList -match "Ubuntu-22.04") {
        Write-Host "  Ubuntu-22.04 gia installato." -ForegroundColor Green
        $wslInstalled = $true
    }
} catch {}

if (-not $wslInstalled) {
    Write-Host "  Ubuntu-22.04 non trovato. Installo..." -ForegroundColor Yellow
    wsl --install -d Ubuntu-22.04
    Write-Host "  Ubuntu-22.04 installato. Se ha chiesto di riavviare, riavvia e rilancia questo script." -ForegroundColor Green
    Write-Host "  Se non ha chiesto riavvio, continuo..." -ForegroundColor Yellow
}

# Verifica che Ubuntu sia raggiungibile
$ubuntuCheck = wsl -d Ubuntu-22.04 -- whoami 2>$null
if (-not $ubuntuCheck) {
    Write-Host "  ERRORE: Ubuntu-22.04 non e raggiungibile." -ForegroundColor Red
    Write-Host "  Riavvia Windows se WSL2 e stato appena installato, poi rilancia." -ForegroundColor Red
    exit 1
}
Write-Host "  Ubuntu-22.04 OK (user: $ubuntuCheck)" -ForegroundColor Green

# --- 2. Verifica GPU ---
Write-Host ""
Write-Host "[2/4] Verifica GPU da WSL2..." -ForegroundColor Yellow
$gpuCheck = wsl -d Ubuntu-22.04 -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
if ($gpuCheck) {
    Write-Host "  GPU: $gpuCheck" -ForegroundColor Green
} else {
    Write-Host "  ATTENZIONE: nvidia-smi non visibile da WSL2." -ForegroundColor Red
    Write-Host "  Aggiorna il driver NVIDIA su Windows (Windows Update o sito NVIDIA)." -ForegroundColor Red
    Write-Host "  Puoi continuare ma vLLM non funzionera senza GPU." -ForegroundColor Yellow
}

# --- 3. Setup dentro Ubuntu ---
Write-Host ""
Write-Host "[3/4] Setup dentro Ubuntu-22.04..." -ForegroundColor Yellow
Write-Host "  (apt, venv, pip, download modelli ~22GB — ci va un po')" -ForegroundColor Yellow
Write-Host ""

$setupScript = @'
#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"

echo "  [3a] Dipendenze sistema..."
sudo apt update -y
sudo apt install -y build-essential python3 python3-venv python3-pip git curl wget

echo ""
echo "  [3b] Repo..."
REPO_DIR="$HOME/shadowgrid-pdf-to-json"
if [ -d "$REPO_DIR" ]; then
  cd "$REPO_DIR"
  git pull --rebase || echo "  git pull fallito, continuo"
else
  git clone https://github.com/Nocco-Matteo/shadowgrid-pdf-to-json.git "$REPO_DIR"
  cd "$REPO_DIR"
fi

echo ""
echo "  [3c] Ambiente Python..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# Scommenta i pesanti CUDA
sed -i 's/^# vllm>=/vllm>=/' requirements.txt
sed -i 's/^# paddleocr>=/paddleocr>=/' requirements.txt
sed -i 's/^# paddlepaddle-gpu>=/paddlepaddle-gpu>=/' requirements.txt
pip install -r requirements.txt

echo ""
echo "  [3d] Download modelli HuggingFace (~22GB)..."
echo "  Non interrompere. Se hai gia i modelli, salta automaticamente."
echo ""
echo "    PaddleOCR-VL-1.6 (~3GB)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('PaddlePaddle/PaddleOCR-VL-1.6')" 2>/dev/null && echo "    OK" || echo "    FALLITO (vLLM lo scarichera al primo run)"
echo "    DeepSeek-OCR-2 (~4GB)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR-2')" 2>/dev/null && echo "    OK" || echo "    FALLITO (vLLM lo scarichera al primo run)"
echo "    Qwen3.8-27B-AWQ (~18GB) — il grosso, pazienza..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nicosuter/Qwen3.8-27B-AWQ')" 2>/dev/null && echo "    OK" || echo "    FALLITO (vLLM lo scarichera al primo run)"

echo ""
echo "  [3e] PDF..."
mkdir -p data/pdfs
PDF_FOUND=0
for dir in "/mnt/c" "/mnt/d" "/mnt/e"; do
  if [ -d "$dir" ]; then
    FOUND=$(find "$dir" -maxdepth 6 -iname "*.pdf" 2>/dev/null | grep -iE "dnd|d&d|player.*handbook|dungeon.*master|monster|dragon|giant|mimic|mordenkainen|tome|fizban|bigby" | head -20)
    if [ -n "$FOUND" ]; then
      echo "    Trovati PDF:"
      echo "$FOUND"
      echo "$FOUND" | while read -r f; do
        cp "$f" data/pdfs/ 2>/dev/null && echo "    copiato: $(basename "$f")" || echo "    fallito: $(basename "$f")"
      done
      PDF_FOUND=1
      break
    fi
  fi
done
if [ "$PDF_FOUND" -eq 0 ]; then
  echo "    PDF non trovati. Copiali a mano:"
  echo "    cp /mnt/c/percorso/*.pdf ~/shadowgrid-pdf-to-json/data/pdfs/"
fi

echo ""
echo "  [3f] Verifica..."
cd "$REPO_DIR"
source .venv/bin/activate
pytest -q 2>&1 | tail -3
ruff check src tests 2>&1 | tail -1

echo ""
echo "============================================"
echo "  SETUP COMPLETATO"
echo "============================================"
echo ""
echo "  Per usare la pipeline, entra in WSL2:"
echo "    wsl -d Ubuntu-22.04"
echo "    cd ~/shadowgrid-pdf-to-json"
echo "    source .venv/bin/activate"
echo "    python -m pipeline.cli run data/pdfs/NOME_FILE.pdf"
echo ""
'@

# Scrivi lo script in una temp dentro Ubuntu e lancialo
$tempScript = "/tmp/setup_inner.sh"
wsl -d Ubuntu-22.04 -- bash -c "cat > $tempScript << 'ENDOFSCRIPT'
$setupScript
ENDOFSCRIPT
chmod +x $tempScript
bash $tempScript"

# --- 4. Fine ---
Write-Host ""
Write-Host "[4/4] Finito." -ForegroundColor Green
Write-Host ""
Write-Host "Per usare la pipeline:" -ForegroundColor Cyan
Write-Host "  wsl -d Ubuntu-22.04" -ForegroundColor White
Write-Host "  cd ~/shadowgrid-pdf-to-json" -ForegroundColor White
Write-Host "  source .venv/bin/activate" -ForegroundColor White
Write-Host "  python -m pipeline.cli run data/pdfs/NOME_FILE.pdf" -ForegroundColor White
Write-Host ""
