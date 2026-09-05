#!/bin/bash
# setup_wsl.sh — Setup completo della pipeline su WSL2 Ubuntu.
# Salva questo file in ~ e lancialo: bash setup_wsl.sh
# Se non hai sudo/apt, non sei in Ubuntu-22.04. Vedi README_WSL.md.
set -e

echo "============================================"
echo "  Shadow Grid PDF-to-JSON — Setup WSL2"
echo "============================================"
echo ""

# 0. PATH (huggingface-cli finisce in ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
echo "[PATH] Aggiunto ~/.local/bin"

# 1. Dipendenze sistema
echo ""
echo "[1/6] Dipendenze di sistema..."
if command -v sudo &>/dev/null; then
  sudo apt update -y
  sudo apt install -y build-essential python3 python3-venv python3-pip git curl
else
  echo "  ATTENZIONE: sudo non disponibile. Se sei in Docker Desktop WSL,"
  echo "  devi installare Ubuntu-22.04: wsl --install -d Ubuntu-22.04"
  echo "  (da PowerShell, non da qui)"
  exit 1
fi

# 2. Verifica GPU
echo ""
echo "[2/6] Verifica GPU..."
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "  GPU OK."
else
  echo "  ATTENZIONE: nvidia-smi non trovato."
  echo "  Aggiorna il driver NVIDIA su Windows (Windows Update o sito NVIDIA)."
  echo "  Il driver Windows fornisce CUDA a WSL2."
  echo "  Puoi continuare, ma vLLM non funzionerà senza GPU."
fi

# 3. Clona o aggiorna la repo
echo ""
echo "[3/6] Repo..."
REPO_DIR="$HOME/shadowgrid-pdf-to-json"
if [ -d "$REPO_DIR" ]; then
  echo "  Repo esistente in $REPO_DIR, aggiorno..."
  cd "$REPO_DIR"
  git pull --rebase || echo "  git pull fallito, continuo con quello che c'è"
else
  git clone https://github.com/Nocco-Matteo/shadowgrid-pdf-to-json.git "$REPO_DIR"
  cd "$REPO_DIR"
fi

# 4. Venv + dipendenze Python
echo ""
echo "[4/6] Ambiente Python..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# Scommenta i pesanti CUDA in requirements.txt
sed -i 's/^# vllm>=/vllm>=/' requirements.txt
sed -i 's/^# paddleocr>=/paddleocr>=/' requirements.txt
sed -i 's/^# paddlepaddle-gpu>=/paddlepaddle-gpu>=/' requirements.txt
echo "  Dipendenze CUDA scommentate in requirements.txt"

pip install -r requirements.txt
echo "  Dipendenze installate."

# 5. Download modelli (~22GB totali, ci va un po')
echo ""
echo "[5/6] Download modelli HuggingFace (~22GB totali)..."
echo "  Questo passo ci mette. Non interromperlo."
echo "  Se hai già i modelli in cache, salta automaticamente."
echo ""

echo "  [5a] PaddleOCR-VL-1.6 (~3GB)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('PaddlePaddle/PaddleOCR-VL-1.6')" 2>/dev/null \
  && echo "  OK" || echo "  FALLITO — vLLM lo scaricherà al primo run comunque"

echo "  [5b] DeepSeek-OCR-2 (~4GB)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR-2')" 2>/dev/null \
  && echo "  OK" || echo "  FALLITO — vLLM lo scaricherà al primo run comunque"

echo "  [5c] Qwen3.8-27B-AWQ (~18GB) — questo è il grosso, pazienza..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nicosuter/Qwen3.8-27B-AWQ')" 2>/dev/null \
  && echo "  OK" || echo "  FALLITO — vLLM lo scaricherà al primo run comunque"

# 6. PDF — cerca su /mnt/c/ e copia
echo ""
echo "[6/6] PDF..."
mkdir -p data/pdfs

# Prova a trovare i PDF su Windows
PDF_FOUND=0
for dir in "/mnt/c/Users" "/mnt/d" "/mnt/e"; do
  if [ -d "$dir" ]; then
    FOUND=$(find "$dir" -maxdepth 5 -iname "*.pdf" -path "*dnd*" 2>/dev/null | head -20)
    if [ -n "$FOUND" ]; then
      echo "  Trovati PDF in $dir:"
      echo "$FOUND"
      echo ""
      echo "  Li copio in data/pdfs/..."
      echo "$FOUND" | while read -r f; do
        cp "$f" data/pdfs/ 2>/dev/null && echo "    copiato: $(basename "$f")" || echo "    fallito: $(basename "$f")"
      done
      PDF_FOUND=1
      break
    fi
  fi
done

if [ "$PDF_FOUND" -eq 0 ]; then
  echo "  PDF non trovati automaticamente su /mnt/c/."
  echo "  Copiali a mano in data/pdfs/:"
  echo "    cp /mnt/c/percorso/dei/tuoi/pdf/*.pdf data/pdfs/"
fi

echo ""
echo "============================================"
echo "  SETUP COMPLETATO"
echo "============================================"
echo ""
echo "  Per usare la pipeline:"
echo "    cd $REPO_DIR"
echo "    source .venv/bin/activate"
echo "    python -m pipeline.cli run data/pdfs/NOME_FILE.pdf"
echo ""
echo "  Per fase singola:"
echo "    python -m pipeline.cli ingest data/pdfs/NOME_FILE.pdf"
echo "    python -m pipeline.cli rasterize data/pdfs/NOME_FILE.pdf"
echo "    python -m pipeline.cli ocr-a data/pdfs/NOME_FILE.pdf"
echo "    python -m pipeline.cli ocr-b data/pdfs/NOME_FILE.pdf"
echo "    python -m pipeline.cli extract data/pdfs/NOME_FILE.pdf"
echo "    python -m pipeline.cli validate data/pdfs/NOME_FILE.pdf"
echo ""
echo "  Test:"
echo "    pytest -q"
echo "    ruff check src tests"
echo ""
