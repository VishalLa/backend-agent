#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Environment variables for Ollama configuration
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-10m}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-false}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_NUM_THREADS="${OLLAMA_NUM_THREADS:-}"
export AGENT_PROVIDER="${AGENT_PROVIDER:-local}"
export AGENT_ENABLE_OLLAMA_FALLBACK="${AGENT_ENABLE_OLLAMA_FALLBACK:-true}"
export OLLAMA_USE_KQUANT="${OLLAMA_USE_KQUANT:-false}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama is not installed or not on PATH." >&2
    exit 1
fi

if ! curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
    echo "ERROR: Ollama is not reachable at ${OLLAMA_BASE_URL}." >&2
    echo "Start the service first, e.g.: ollama serve" >&2
    exit 1
fi

echo "Ollama local runtime config"
echo "  base_url: ${OLLAMA_BASE_URL}"
echo "  keep_alive: ${OLLAMA_KEEP_ALIVE}"
echo "  flash_attention: ${OLLAMA_FLASH_ATTENTION}"
echo "  kv_cache_type: ${OLLAMA_KV_CACHE_TYPE}"
echo "  num_threads: ${OLLAMA_NUM_THREADS:-auto}"
echo "  use_kquant: ${OLLAMA_USE_KQUANT}"

echo

echo "Checking local models..."
python3 - <<'PY'
import os
from config import Config
cfg = Config.from_env()
print(f"provider={cfg.provider}")
print(f"ollama_model={cfg.ollama_model}")
print(f"ollama_summary_model={cfg.ollama_summary_model}")
print(f"ollama_use_kquant={cfg.ollama_use_kquant}")
print(f"keep_alive={cfg.ollama_keep_alive}")
print(f"flash_attention={cfg.ollama_flash_attention}")
print(f"kv_cache_type={cfg.ollama_kv_cache_type}")
print(f"num_thread={cfg.ollama_num_thread}")
PY

echo

if [ "${OLLAMA_USE_KQUANT}" = "true" ] || [ "${OLLAMA_USE_KQUANT}" = "1" ]; then
    echo "Pulling K-quant model..."
    ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M || echo "Note: Model pull attempted; check Ollama logs if needed."
else
    echo "Using default model: MFDoom/deepseek-coder-v2-tool-calling:16b"
    echo "To switch to K-quant model, set: OLLAMA_USE_KQUANT=true"
fi

echo "Use the app normally via: python3 main.py"
