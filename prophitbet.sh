#!/bin/bash
# Navigate to the directory of the script
cd "$(dirname "$0")"

# Activate the ProphitBet virtual environment (Python 3.11).
# Override its location with PROPHITBET_VENV (default: ~/prophitbet-venv).
source "${PROPHITBET_VENV:-$HOME/prophitbet-venv}/bin/activate"

# Run main.py
python main.py
