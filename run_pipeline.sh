#!/bin/zsh
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR" || exit 1

.venv/bin/python3 daily_pipeline.py >> logs/run.log 2>&1
