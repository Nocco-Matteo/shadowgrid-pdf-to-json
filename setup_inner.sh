#!/bin/bash
# setup_inner.sh — lanciato da setup_windows.ps1 dentro Ubuntu-22.04.
# Non lanciarlo a mano; usa setup_windows.ps1 da PowerShell.
# Argomento $1 (opzionale): path WSL del repo su Windows, es. /mnt/c/.../repo.
# Se presente, i sorgenti vengono copiati da lì (stessa versione del tuo checkout);
# altrimenti fallback a git clone da GitHub.
#
# -e -u -o pipefail: qualunque fallimento (anche dentro pipe, es. pytest | tail)
# termina lo script con codice non-zero e setup_windows.ps1 si ferma.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

echo "  [3a] Dipendenze sistema..."
sudo apt update -y
sudo apt install -y build-essential git curl wget rsync \
  software-properties-common

# Ubuntu 22.04 (jammy) ships Python 3.10 di default, ma il progetto richiede
# >= 3.11 (pyproject.toml). Installa esplicitamente 3.11 dai deadsnakes PPA.
echo ""
echo "  [3a-bis] Python 3.11 (il default di Ubuntu 22.04 è 3.10, troppo vecchio)..."
if ! command -v python3.11 >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update -y
  sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi
PY="$(command -v python3.11)"
echo "    Interprete: $PY ($("$PY" --version 2>&1))"

echo ""
echo "  [3b] Repo..."
REPO_DIR="$HOME/shadowgrid-pdf-to-json"
SRC_DIR="${1:-}"
if [ -n "$SRC_DIR" ] && [ -d "$SRC_DIR" ]; then
  echo "    Copia da checkout Windows: $SRC_DIR"
  mkdir -p "$REPO_DIR"
  rsync -a --delete \
    --exclude '.venv' --exclude '.git' --exclude 'runs' \
    --exclude '__pycache__' --exclude '*.egg-info' --exclude '.pytest_cache' \
    "$SRC_DIR/" "$REPO_DIR/"
  cd "$REPO_DIR"
elif [ -d "$REPO_DIR" ]; then
  cd "$REPO_DIR"
  git pull --rebase || echo "  git pull fallito, continuo"
else
  git clone https://github.com/Nocco-Matteo/shadowgrid-pdf-to-json.git "$REPO_DIR"
  cd "$REPO_DIR"
fi

echo ""
echo "  [3c] Ambiente Python ($PY)..."
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
  echo "    Riuso venv esistente ($(.venv/bin/python --version 2>&1))"
else
  if [ -d ".venv" ]; then
    echo "    venv esistente con Python troppo vecchio ($(.venv/bin/python --version 2>&1)): lo ricreo"
    rm -rf .venv
  fi
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel

# Scommenta i pesanti CUDA
sed -i 's/^# vllm>=/vllm>=/' requirements.txt
sed -i 's/^# paddleocr\[/paddleocr[/' requirements.txt
sed -i 's/^# paddlepaddle>=/paddlepaddle>=/' requirements.txt
# Un paddlepaddle-gpu rimasto da un vecchio setup sovrascriverebbe `paddle`
# (2.x, incompatibile con paddlex 3.x). Disinstallarlo lascia il modulo
# `paddle` senza file: la CPU va reinstallata a forza.
if pip show paddlepaddle-gpu >/dev/null 2>&1; then
  pip uninstall -y paddlepaddle-gpu
  pip install --force-reinstall --no-deps "paddlepaddle>=3.2"
fi
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
    # || true: find/grep possono ricevere SIGPIPE da head; un output vuoto
    # (grep senza match) non deve abortire lo script.
    FOUND=$(find "$dir" -maxdepth 6 -iname "*.pdf" 2>/dev/null | grep -iE "dnd|d&d|player.*handbook|dungeon.*master|monster|dragon|giant|mimic|mordenkainen|tome|fizban|bigby" | head -20 || true)
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
# con pipefail: se pytest/ruff falliscono, lo script esce non-zero
pytest -q 2>&1 | tail -3
ruff check src tests 2>&1 | tail -1

echo ""
echo "  SETUP COMPLETATO."
