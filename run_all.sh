#!/usr/bin/env bash
set -euo pipefail

# Create venv and install pinned requirements
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  pip install numpy scipy matplotlib pytest
fi

# Run tests
pytest -q

# Run the full simulation (short sample run)
python src/full_monte.py --grid 50 --steps 5 --seed 7 --c 2.0 --alpha 0.1

# Build arXiv PDF if present
if [ -d arxiv ]; then
  pushd arxiv
  if command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode main.tex || true
    pdflatex -interaction=nonstopmode main.tex || true
  fi
  popd
fi

echo "run_all complete"
