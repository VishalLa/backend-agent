#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Parse CLI arguments
MODE="${1:-full}"
AGENT="${2:-}"

CMD_SANDBOX="cd '$PROJECT_DIR' && bash sandbox/run.sh"
CMD_CELERY="cd '$PROJECT_DIR' && celery -A database.service.celery_app worker --loglevel=info"
CMD_CLI="cd '$PROJECT_DIR' && python3 main.py"
if [[ -n "$AGENT" ]]; then
    CMD_CLI="$CMD_CLI --agent $AGENT"
fi

# Print usage
print_usage() {
    cat << EOF
Usage: ./run.sh [MODE] [AGENT]

Modes:
  cli           Run agent CLI only (no services) - DEFAULT
  full          Launch full stack (sandbox + celery + CLI)
  sandbox-only  Launch only sandbox service
  celery-only   Launch only celery worker
  dev           Launch CLI with services in background

Agents (for CLI mode):
  backend       Flask/FastAPI, business logic, integrations
  ml            Training, evaluation, data pipelines
  git           Version control, branches, commits, push
  algorithms    Correctness- and complexity-sensitive code
  (if not specified, you'll be prompted to choose)

Examples:
  ./run.sh cli backend        # Run CLI with backend agent
  ./run.sh full               # Full stack with interactive agent selection
  ./run.sh dev ml              # CLI (ml) + services backgrounded
EOF
}

case "${MODE}" in
    cli)
        echo "=========================================="
        echo " Starting Coding Agent CLI               "
        echo "=========================================="
        exec python3 main.py ${AGENT:+--agent $AGENT}
        ;;
    full)
        echo "=========================================="
        echo " Launching Multi-Agent System Services   "
        echo "=========================================="
        echo "1. Sandbox: $CMD_SANDBOX"
        echo "2. Celery:  $CMD_CELERY"
        echo "3. CLI:     $CMD_CLI"
        echo "=========================================="
        ;;
    sandbox-only)
        echo "=========================================="
        echo " Launching Sandbox Service Only          "
        echo "=========================================="
        exec bash sandbox/run.sh
        ;;
    celery-only)
        echo "=========================================="
        echo " Launching Celery Worker Only            "
        echo "=========================================="
        exec celery -A database.service.celery_app worker --loglevel=info
        ;;
    dev)
        echo "=========================================="
        echo " Launching Dev Mode (CLI + Services)     "
        echo "=========================================="
        echo "1. Sandbox:  Running in background"
        echo "2. Celery:   Running in background"
        echo "3. CLI:      $CMD_CLI"
        echo "=========================================="
        bash sandbox/run.sh > /tmp/sandbox.log 2>&1 &
        SANDBOX_PID=$!
        celery -A database.service.celery_app worker --loglevel=info > /tmp/celery.log 2>&1 &
        CELERY_PID=$!
        echo "Sandbox PID: $SANDBOX_PID (logs: /tmp/sandbox.log)"
        echo "Celery PID:  $CELERY_PID (logs: /tmp/celery.log)"
        trap "kill $SANDBOX_PID $CELERY_PID 2>/dev/null || true" EXIT
        exec python3 main.py ${AGENT:+--agent $AGENT}
        ;;
    help|--help|-h)
        print_usage
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE"
        print_usage
        exit 1
        ;;
esac

if [[ "$MODE" != "full" ]]; then
    exit 0
fi

# Platform-specific terminal launching for full mode
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - AppleScript
    osascript -e "tell app \"Terminal\" to do script \"$CMD_SANDBOX\""
    osascript -e "tell app \"Terminal\" to do script \"$CMD_CELERY\""
    osascript -e "tell app \"Terminal\" to do script \"$CMD_CLI\""
    echo "Launched 3 separate Terminal windows."

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - Terminal Emulators
    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal --title="Sandbox Service" -- bash -c "$CMD_SANDBOX; exec bash"
        gnome-terminal --title="Celery Worker" -- bash -c "$CMD_CELERY; exec bash"
        gnome-terminal --title="Coding Agent CLI" -- bash -c "$CMD_CLI; exec bash"
        echo "Launched 3 gnome-terminal windows."
    elif command -v konsole &>/dev/null; then
        konsole --new-tab -e bash -c "$CMD_SANDBOX; exec bash" &
        konsole --new-tab -e bash -c "$CMD_CELERY; exec bash" &
        konsole --new-tab -e bash -c "$CMD_CLI; exec bash" &
        echo "Launched 3 Konsole tabs."
    elif command -v xterm &>/dev/null; then
        xterm -T "Sandbox" -e bash -c "$CMD_SANDBOX; exec bash" &
        xterm -T "Celery" -e bash -c "$CMD_CELERY; exec bash" &
        xterm -T "Coding Agent CLI" -e bash -c "$CMD_CLI; exec bash" &
        echo "Launched 3 xterm windows."
    elif command -v tmux &>/dev/null; then
        tmux new-session -d -s multiagent "$CMD_SANDBOX"
        tmux split-window -t multiagent "$CMD_CELERY"
        tmux split-window -t multiagent "$CMD_CLI"
        tmux select-layout -t multiagent tiled
        tmux attach-session -t multiagent
    else
        echo "No supported terminal emulator found. Please run the commands manually."
    fi

elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows Git Bash
    start bash -c "$CMD_SANDBOX; exec bash"
    start bash -c "$CMD_CELERY; exec bash"
    start bash -c "$CMD_CLI; exec bash"
    echo "Launched 3 separate windows on Windows."

else
    echo "Unsupported OS environment ($OSTYPE). Please start services manually."
fi
