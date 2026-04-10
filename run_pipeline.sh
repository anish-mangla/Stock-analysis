#!/bin/zsh
cd /Users/anish/Files/stock_temp/Stock-analysis || exit 1

# Optional: activate virtual environment
source .venv/bin/activate

python3 main.py >> logs/run.log 2>&1