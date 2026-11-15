#!/bin/bash
# Navigate to the directory of the script
cd "$(dirname "$0")"

# Activate the ProphitBet virtual environment (Python 3.11)
source "$HOME/prophitbet-venv/bin/activate"

# Run main.py
python main.py
