#!/usr/bin/env bash
# test.sh — start required Docker services, run the full test suite, then tear down.
#
# Usage:
#   ./test.sh              # run all tests (unit + integration)
#   ./test.sh unit         # skip integration tests (no Docker needed)
#   ./test.sh integration  # run integration tests only

set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────────────
COMPOSE_FILE="docker-compose.yml"
SERVICES="redis chromadb"        # Ollama is optional and heavy; excluded by default
REDIS_HOST="127.0.0.1"
REDIS_PORT="6379"
CHROMA_HOST="127.0.0.1"
CHROMA_PORT="8000"
WAIT_TIMEOUT=60                  # seconds to wait for each service to become healthy

# ── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
info()  { echo -e "${GREEN}[test.sh]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[test.sh]${RESET} $*"; }
error() { echo -e "${RED}[test.sh] ERROR:${RESET} $*" >&2; }

# ── Cleanup trap ─────────────────────────────────────────────────────────────
MODE="${1:-all}"
DOCKER_STARTED=false

cleanup() {
    if [[ "$DOCKER_STARTED" == "true" ]]; then
        info "Stopping Docker services..."
        docker compose -f "$COMPOSE_FILE" stop $SERVICES >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# ── Helpers ──────────────────────────────────────────────────────────────────
wait_for_tcp() {
    local name="$1" host="$2" port="$3"
    local elapsed=0
    info "Waiting for $name ($host:$port)..."
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if (( elapsed >= WAIT_TIMEOUT )); then
            error "$name did not become reachable on $host:$port within ${WAIT_TIMEOUT}s — aborting."
            exit 1
        fi
        sleep 1
        (( elapsed++ ))
    done
    info "$name is up (${elapsed}s)."
}

wait_for_redis() {
    local elapsed=0
    info "Waiting for Redis to respond to PING..."
    while ! docker compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; do
        if (( elapsed >= WAIT_TIMEOUT )); then
            error "Redis did not respond to PING within ${WAIT_TIMEOUT}s — aborting."
            exit 1
        fi
        sleep 1
        (( elapsed++ ))
    done
    info "Redis ready (${elapsed}s)."
}

wait_for_chroma() {
    local elapsed=0
    info "Waiting for ChromaDB /api/v1/heartbeat..."
    while ! curl -sf "http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/heartbeat" >/dev/null 2>&1; do
        if (( elapsed >= WAIT_TIMEOUT )); then
            error "ChromaDB did not respond within ${WAIT_TIMEOUT}s — aborting."
            exit 1
        fi
        sleep 1
        (( elapsed++ ))
    done
    info "ChromaDB ready (${elapsed}s)."
}

check_docker() {
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running. Please start Docker and retry."
        exit 1
    fi
    if ! command -v docker compose >/dev/null 2>&1; then
        error "'docker compose' (v2) not found. Please install Docker Compose v2."
        exit 1
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
case "$MODE" in
    unit)
        info "Running unit tests only (no Docker required)..."
        exec python -m pytest -m "not integration" --tb=short "$@"
        ;;
    integration)
        PYTEST_MARKS="-m integration"
        ;;
    all)
        PYTEST_MARKS=""
        ;;
    *)
        error "Unknown mode '$MODE'. Use: all | unit | integration"
        exit 1
        ;;
esac

# Start Docker services
check_docker
info "Starting Docker services: $SERVICES"
docker compose -f "$COMPOSE_FILE" up -d $SERVICES
DOCKER_STARTED=true

# Wait for each service to be healthy — FAIL if they don't come up
wait_for_redis
wait_for_chroma

# Activate virtualenv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Run pytest
info "Running tests (mode=$MODE)..."
# shellcheck disable=SC2086
python -m pytest $PYTEST_MARKS --tb=short "$@"
