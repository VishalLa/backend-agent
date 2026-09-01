#!/usr/bin/env bash
set -euo pipefail

# Phase 8: K-quant Upgrade Validation Script
#
# This script compares the Phase 7 tool-calling model with the Phase 8 K-quant variant
# using the eval harness to determine if the K-quant model offers better accuracy
# at the same speed/size.
#
# Usage:
#   bash phase8_validate.sh [--run-only-phase7] [--run-only-kquant]
#
# Output:
#   - eval_results_phase7.json (Phase 7 tool-calling baseline)
#   - eval_results_kquant.json (Phase 8 K-quant results)
#   - phase8_comparison.txt (side-by-side comparison)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="${PROJECT_DIR}/eval_results"
mkdir -p "$RESULTS_DIR"

PHASE7_RESULT="$RESULTS_DIR/phase7_toolcalling_results.json"
KQUANT_RESULT="$RESULTS_DIR/phase8_kquant_results.json"
COMPARISON_RESULT="$RESULTS_DIR/phase8_comparison.txt"

RUN_PHASE7=true
RUN_KQUANT=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-only-phase7)
            RUN_KQUANT=false
            shift
            ;;
        --run-only-kquant)
            RUN_PHASE7=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--run-only-phase7] [--run-only-kquant]"
            exit 1
            ;;
    esac
done

if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama is not installed." >&2
    exit 1
fi

echo "Phase 8: K-quant Validation"
echo "==========================="
echo

# Check if eval harness is available
if [ ! -f "$PROJECT_DIR/agent/eval.py" ]; then
    echo "ERROR: eval.py not found at $PROJECT_DIR/agent/eval.py" >&2
    echo "Phase 4 (eval harness) is a prerequisite for Phase 8 validation." >&2
    exit 1
fi

# Phase 7: Run eval with tool-calling model
if [ "$RUN_PHASE7" = "true" ]; then
    echo "Phase 7: Running eval harness with tool-calling model..."
    echo "  Model: MFDoom/deepseek-coder-v2-tool-calling:16b"
    echo

    OLLAMA_USE_KQUANT=false PYTHONPATH="$PROJECT_DIR" python3 - <<'PY' > "$PHASE7_RESULT" 2> "${PHASE7_RESULT}.err"
import sys
import json
from pathlib import Path

from config import Config
from agent.eval import EvalHarness

cfg = Config.from_env()
print(f"[Phase 7] Model: {cfg.backend_model_name}", file=sys.stderr)
results = EvalHarness(cfg).run()
print(json.dumps({
    "phase": "phase7_toolcalling", "model": cfg.backend_model_name,
    "passed": results["passed"], "failed": results["failed"],
    "invocation_failures": results["invocation_failures"],
    "avg_latency_seconds": results["avg_latency"], "results": results,
}))
PY

    if [ $? -eq 0 ] && [ -s "$PHASE7_RESULT" ]; then
        echo "✓ Phase 7 eval completed. Results saved to $PHASE7_RESULT"
    else
        echo "✗ Phase 7 eval failed or produced no output."
        echo "  Check that Ollama is running with the tool-calling model pulled."
    fi
    echo
fi

# Phase 8: Run eval with K-quant model
if [ "$RUN_KQUANT" = "true" ]; then
    echo "Phase 8: Running eval harness with K-quant model..."
    echo "  Model: deepseek-coder-v2:16b-lite-instruct-q4_K_M"
    echo

    # Pull the K-quant model first
    echo "Pulling K-quant model (if not already present)..."
    ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M || echo "Note: K-quant pull may have failed; check Ollama logs."
    echo

    OLLAMA_USE_KQUANT=true PYTHONPATH="$PROJECT_DIR" python3 - <<'PY' > "$KQUANT_RESULT" 2> "${KQUANT_RESULT}.err"
import sys
import json
from pathlib import Path

from config import Config
from agent.eval import EvalHarness

cfg = Config.from_env()
print(f"[Phase 8] Model: {cfg.backend_model_name}", file=sys.stderr)
results = EvalHarness(cfg).run()
print(json.dumps({
    "phase": "phase8_kquant", "model": cfg.backend_model_name,
    "passed": results["passed"], "failed": results["failed"],
    "invocation_failures": results["invocation_failures"],
    "avg_latency_seconds": results["avg_latency"], "results": results,
}))
PY

    if [ $? -eq 0 ] && [ -s "$KQUANT_RESULT" ]; then
        echo "✓ Phase 8 eval completed. Results saved to $KQUANT_RESULT"
    else
        echo "✗ Phase 8 eval failed or produced no output."
        echo "  Ensure the K-quant model is available on Ollama."
    fi
    echo
fi

# Generate comparison report
echo "Generating comparison report..."
if [ -f "$PHASE7_RESULT" ] && [ -f "$KQUANT_RESULT" ]; then
    python3 - <<'PY' > "$COMPARISON_RESULT"
import json
import sys

try:
    with open("eval_results/phase7_toolcalling_results.json") as f:
        phase7 = json.load(f)
    with open("eval_results/phase8_kquant_results.json") as f:
        phase8 = json.load(f)

    p7_model = phase7.get("model", "unknown")
    p8_model = phase8.get("model", "unknown")
    p7_passed = phase7.get("passed", 0)
    p8_passed = phase8.get("passed", 0)
    p7_invocation_failures = phase7.get("invocation_failures", 0)
    p8_invocation_failures = phase8.get("invocation_failures", 0)
    p7_latency = phase7.get("avg_latency_seconds", 0)
    p8_latency = phase8.get("avg_latency_seconds", 0)
    latency_percent = ((p8_latency / p7_latency - 1) * 100) if p7_latency else None

    report = f"""
Phase 8 K-quant Upgrade Validation Report
==========================================

Phase 7 (Tool-calling model):
  Model: {p7_model}
  Tests Passed: {p7_passed}
  Invocation Failures: {p7_invocation_failures}
  Avg Latency: {p7_latency:.2f}s

Phase 8 (K-quant model):
  Model: {p8_model}
  Tests Passed: {p8_passed}
  Invocation Failures: {p8_invocation_failures}
  Avg Latency: {p8_latency:.2f}s

Comparison:
  Accuracy Gain: {p8_passed - p7_passed:+d} tests
  Latency Change: {p8_latency - p7_latency:+.2f}s ({f'{latency_percent:+.1f}%' if latency_percent is not None else 'n/a'})

Recommendation:
"""

    if p7_invocation_failures or p8_invocation_failures:
        report += "  ! Benchmark is invalid: one or both models could not complete all invocations. Check the .err files and Ollama model availability."
    elif p8_passed > p7_passed and p8_latency <= p7_latency * 1.1:
        report += f"  ✓ K-quant model is better: +{p8_passed - p7_passed} accuracy with {max(0, p7_latency - p8_latency):.2f}s speedup or minimal overhead."
        report += "\n  Action: Set OLLAMA_USE_KQUANT=true in production."
    elif p8_passed >= p7_passed and p8_latency <= p7_latency * 1.05:
        report += f"  ~ K-quant model is comparable with minimal latency overhead."
        report += "\n  Action: Consider upgrading if accuracy is important for your use case."
    else:
        report += f"  ✗ Tool-calling model is better. Recommended to keep Phase 7 as default."
        report += "\n  Action: Run Phase 7 config is still the default; keep OLLAMA_USE_KQUANT=false."

    print(report)
except Exception as e:
    print(f"ERROR generating comparison: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY

    echo "✓ Comparison report saved to $COMPARISON_RESULT"
    echo
    echo "Results Summary:"
    echo "==============="
    cat "$COMPARISON_RESULT" || true
else
    echo "⚠ Both Phase 7 and Phase 8 results required for comparison."
    echo "  Run both evals or specify --run-only-phase7 or --run-only-kquant."
fi

echo
echo "Detailed results:"
echo "  Phase 7 (tool-calling): $PHASE7_RESULT"
echo "  Phase 8 (K-quant): $KQUANT_RESULT"
echo "  Comparison: $COMPARISON_RESULT"
echo
echo "To switch to K-quant in production after validation:"
echo "  export OLLAMA_USE_KQUANT=true"
echo "  bash run_ollama.sh"
