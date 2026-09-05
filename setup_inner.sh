#!/bin/bash
# setup_inner.sh — lanciato da setup_windows.ps1 dentro Ubuntu-22.04.
# Non lanciarlo a mano; usa setup_windows.ps1 da PowerShell.
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
echo "  Non interrompere."
echo ""
echo "    PaddleOCR-VL-1.6 (~3GB)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('PaddlePaddle/PaddleOCR-VL-1.6')" 2>/dev/null && echo "    OK" || echo "    FALLITO - vLLM lo scarichera al primo run"
echo "    DeepSeek-OCR-2 (~4GB)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR-2')" 2>/dev/null && echo "    OK" || echo "    FALLITO - vLLM lo scarichera al primo run"
echo "    Qwen3.8-27B-AWQ (~18GB) - il grosso..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nicosuter/Qwen3.8-27B-AWQ')" 2>/dev/null && echo "    OK" || echo "    FALLITO - vLLM lo scarichera al primo run"

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
echo "  SETUP COMPLETATO."
