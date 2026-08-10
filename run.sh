#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CMD_SANDBOX="cd '$PROJECT_DIR' && bash sandbox/run.sh"
CMD_CELERY="cd '$PROJECT_DIR' && celery -A database.service.celery_app worker --loglevel=info"
CMD_UI="cd '$PROJECT_DIR' && streamlit run main.py"

echo "=========================================="
echo " Launching Multi-Agent System Services   "
echo "=========================================="
echo "1. Sandbox: $CMD_SANDBOX"
echo "2. Celery:  $CMD_CELERY"
echo "3. UI:      $CMD_UI"
echo "=========================================="

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - AppleScript
    osascript -e "tell app \"Terminal\" to do script \"$CMD_SANDBOX\""
    osascript -e "tell app \"Terminal\" to do script \"$CMD_CELERY\""
    osascript -e "tell app \"Terminal\" to do script \"$CMD_UI\""
    echo "Launched 3 separate Terminal windows."

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - Terminal Emulators
    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal --title="Sandbox Service" -- bash -c "$CMD_SANDBOX; exec bash"
        gnome-terminal --title="Celery Worker" -- bash -c "$CMD_CELERY; exec bash"
        gnome-terminal --title="Streamlit Dashboard" -- bash -c "$CMD_UI; exec bash"
        echo "Launched 3 gnome-terminal windows."
    elif command -v konsole &>/dev/null; then
        konsole --new-tab -e bash -c "$CMD_SANDBOX; exec bash" &
        konsole --new-tab -e bash -c "$CMD_CELERY; exec bash" &
        konsole --new-tab -e bash -c "$CMD_UI; exec bash" &
        echo "Launched 3 Konsole tabs."
    elif command -v xterm &>/dev/null; then
        xterm -T "Sandbox" -e bash -c "$CMD_SANDBOX; exec bash" &
        xterm -T "Celery" -e bash -c "$CMD_CELERY; exec bash" &
        xterm -T "UI" -e bash -c "$CMD_UI; exec bash" &
        echo "Launched 3 xterm windows."
    elif command -v tmux &>/dev/null; then
        tmux new-session -d -s multiagent "$CMD_SANDBOX"
        tmux split-window -t multiagent "$CMD_CELERY"
        tmux split-window -t multiagent "$CMD_UI"
        tmux select-layout -t multiagent tiled
        tmux attach-session -t multiagent
    else
        echo "No supported terminal emulator found. Please run the commands manually."
    fi

elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows Git Bash
    start bash -c "$CMD_SANDBOX; exec bash"
    start bash -c "$CMD_CELERY; exec bash"
    start bash -c "$CMD_UI; exec bash"
    echo "Launched 3 separate windows on Windows."

else
    echo "Unsupported OS environment ($OSTYPE). Please start services manually."
fi
