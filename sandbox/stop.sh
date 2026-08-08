#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-agent-sandbox}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Stopping and removing ${CONTAINER_NAME} ..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
    echo "Done."
else
    echo "No container named ${CONTAINER_NAME} found."
fi