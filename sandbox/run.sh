#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sandbox:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-agent-sandbox}"
NETWORK_NAME="${NETWORK_NAME:-agent-sandbox-net}"
HOST_PORT="${HOST_PORT:-8787}"
CONTAINER_PORT=8787

MEM_LIMIT="${MEM_LIMIT:-512m}"
PIDS_LIMIT="${PIDS_LIMIT:-128}"
WORKSPACE_TMPFS_MB="${WORKSPACE_TMPFS_MB:-256}"
TMP_TMPFS_MB="${TMP_TMPFS_MB:-256}"

ALLOW_NETWORK="${ALLOW_NETWORK:-0}"   # set to 1 for real internet access (e.g. pip install)

echo "Image:            ${IMAGE}"
echo "Container name:    ${CONTAINER_NAME}"
echo "Host port:          ${HOST_PORT} -> ${CONTAINER_PORT}"
echo "Memory limit:         ${MEM_LIMIT} (no swap)"
echo "PIDs limit:            ${PIDS_LIMIT}"
echo "Network:                $([ "$ALLOW_NETWORK" = "1" ] && echo "ENABLED (not recommended)" || echo "internal only - no external route, host can still reach it via -p")"
echo

# --- clean up any previous instance with this name ---
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Removing existing container named ${CONTAINER_NAME} ..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# --- pick a network ---
#
# IMPORTANT: we deliberately do NOT use `--network none` here. That mode
# strips the container down to a loopback-only interface, which also means
# `-p HOST_PORT:CONTAINER_PORT` has nothing to publish - the host loses the
# ability to reach /health and /exec entirely (this is what broke the first
# version of this script).
#
# Instead we use a custom network created with `--internal`: Docker still
# DNATs the published port from the host straight to the container (that
# happens at the host/daemon level, unaffected by --internal), but the
# network itself has no default route to the outside world, so code running
# inside the sandbox still can't reach the real internet.
if [ "${ALLOW_NETWORK}" = "1" ]; then
    NETWORK_FLAGS=(--network bridge)
else
    if ! docker network inspect "${NETWORK_NAME}" >/dev/null 2>&1; then
        echo "Creating internal (no external route) network ${NETWORK_NAME} ..."
        docker network create --internal "${NETWORK_NAME}" >/dev/null
    fi
    NETWORK_FLAGS=(--network "${NETWORK_NAME}")
fi

docker run -d \
    --name "${CONTAINER_NAME}" \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    "${NETWORK_FLAGS[@]}" \
    --read-only \
    --tmpfs "/tmp:size=${TMP_TMPFS_MB}m,mode=1777" \
    --tmpfs "/workspace:size=${WORKSPACE_TMPFS_MB}m,mode=1777" \
    --memory "${MEM_LIMIT}" \
    --memory-swap "${MEM_LIMIT}" \
    --pids-limit "${PIDS_LIMIT}" \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --user sandbox \
    "${IMAGE}"

echo
echo "Started. Waiting for /health ..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${HOST_PORT}/health" >/dev/null 2>&1; then
        echo "Sandbox is up: http://127.0.0.1:${HOST_PORT}"
        exit 0
    fi
    sleep 0.5
done

echo "Sandbox did not become healthy in time. Recent logs:"
docker logs --tail 50 "${CONTAINER_NAME}"
exit 1
