#!/usr/bin/env bash
# Run the full benchmark (resumable). Paced for free-tier Gemini keys (15 requests/minute per model).
set -e
cd "$(dirname "$0")/.."
export LLM_MIN_INTERVAL=4.2
python3 -m pytest -q tests
python3 -m agr.benchmark run --split public --pipelines agentic,graphrag,rag --workers 3
python3 -m agr.benchmark run --split hidden --pipelines agentic,graphrag,rag --workers 3
python3 -m agr.benchmark judge --split public
python3 -m agr.benchmark report
python3 -m agr.dashboard
