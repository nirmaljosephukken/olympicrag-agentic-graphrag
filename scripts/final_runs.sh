#!/usr/bin/env bash
# Final runs of the three pipelines with the final toolset, all graph queries through the TigerGraph MCP server.
# Resumable (already answered questions are skipped); ordered by priority.
cd "$(dirname "$0")/.."
export LLM_MIN_INTERVAL=${LLM_MIN_INTERVAL:-4.2} TG_TRANSPORT=mcp
python3 -m pytest -q tests || exit 1
for s in hard public hidden time open; do python3 -m agr.benchmark run --split $s --pipelines agentic,graphrag,rag --workers 3; done
for s in public hard time open; do python3 -m agr.benchmark judge --split $s; done
python3 -m agr.benchmark report; python3 -m agr.dashboard; python3 -m agr.submission
