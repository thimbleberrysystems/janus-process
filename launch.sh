#!/usr/bin/env bash
# launch.sh — Bootstrap and start the Janus Process brain system.
#
# What this script does
# ---------------------
#   1. Creates .env from .env.example if one does not exist
#   2. Finds a suitable Python 3.11+ interpreter
#   3. Creates .venv and installs project dependencies if needed
#   4. Verifies Docker is installed and its daemon is running
#   5. Starts redis, chromadb, and ollama via Docker Compose
#   6. Waits for each service to become healthy
#   7. Pulls all required Ollama models (when LLM_PROVIDER=ollama)
#   8. Starts the API server — or the CLI with --cli
#
# Usage
# -----
#   ./launch.sh           start FastAPI server  (http://localhost:8080/docs)
#   ./launch.sh --cli     start interactive CLI loop
#   ./launch.sh --help    show this message

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${BLUE}▸${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
die()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }
hdr()  { echo -e "\n${BOLD}━━  $*${NC}"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-}"
if [[ "$MODE" == "--help" || "$MODE" == "-h" ]]; then
    sed -n '/^# Usage/,/^#$/{ /^# */s/^# \?//p }' "$0"
    exit 0
fi

# ── .env ──────────────────────────────────────────────────────────────────────
hdr "Environment"
if [[ ! -f .env ]]; then
    cp .env.example .env
    warn ".env not found — created from .env.example. Set your API keys before production use."
else
    ok ".env present"
fi

# Load env vars: strip comments and blank lines, remove inline comments
_load_env() {
    set -o allexport
    # shellcheck source=.env
    source <(grep -v '^\s*[#;]' .env | grep -v '^\s*$' | sed 's/[[:space:]]*#.*//' | tr -d '\r')
    set +o allexport
}
_load_env

LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
OLLAMA_HOST="${OLLAMA_URL:-http://127.0.0.1:11434}"
CHROMA_PORT="${CHROMA_PORT:-8000}"

# ── Python ────────────────────────────────────────────────────────────────────
hdr "Python"
PYTHON=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[[ -z "$PYTHON" ]] && die "No Python 3.11+ interpreter found. Install Python and retry."

PY_MAJ=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MIN=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
PY_VER="${PY_MAJ}.${PY_MIN}"

if [[ "$PY_MAJ" -lt 3 || ( "$PY_MAJ" -eq 3 && "$PY_MIN" -lt 11 ) ]]; then
    die "Python 3.11+ required, found $PY_VER at $PYTHON"
fi
ok "Python $PY_VER  ($PYTHON)"

# ── Virtual environment ───────────────────────────────────────────────────────
hdr "Virtual environment"
if [[ ! -d .venv ]]; then
    log "Creating .venv with $PYTHON..."
    "$PYTHON" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
ok ".venv active  ($(python --version))"

log "Installing / syncing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -e '.[dev]'
ok "Dependencies ready"

# ── Docker ────────────────────────────────────────────────────────────────────
hdr "Docker"
if ! command -v docker &>/dev/null; then
    die "Docker not found. Install Docker Desktop (Mac/Win) or Docker Engine (Linux) and retry."
fi
if ! docker info &>/dev/null 2>&1; then
    die "Docker daemon is not running. Start Docker and retry."
fi
ok "Docker daemon running  ($(docker --version | head -1))"

# ── Services ──────────────────────────────────────────────────────────────────
hdr "Services  (redis · chromadb · ollama)"
log "docker compose up -d redis chromadb ollama"
docker compose up -d redis chromadb ollama

# Redis
log "Waiting for Redis..."
for i in $(seq 1 30); do
    docker exec janus-redis redis-cli ping &>/dev/null 2>&1 && { ok "Redis ready"; break; }
    sleep 1
    [[ $i -eq 30 ]] && die "Redis did not become healthy within 30 s"
done

# ChromaDB — pure bash TCP check; no curl/python needed
log "Waiting for ChromaDB..."
for i in $(seq 1 60); do
    if (echo > /dev/tcp/localhost/"${CHROMA_PORT}") 2>/dev/null; then
        ok "ChromaDB ready"
        break
    fi
    sleep 1
    [[ $i -eq 60 ]] && die "ChromaDB did not become healthy within 60 s"
done

# Ollama — pure bash TCP check
log "Waiting for Ollama..."
for i in $(seq 1 90); do
    if (echo > /dev/tcp/localhost/11434) 2>/dev/null; then
        ok "Ollama ready"
        break
    fi
    sleep 1
    [[ $i -eq 90 ]] && die "Ollama did not become healthy within 90 s"
done

# ── Ollama model pulls ────────────────────────────────────────────────────────
if [[ "$LLM_PROVIDER" == "ollama" ]]; then
    hdr "Ollama models"

    # Collect unique model names from env vars
    declare -A _SEEN
    MODELS_TO_PULL=()
    for _var in OLLAMA_MODEL AMYGDALA_MODEL PFC_MODEL BG_MODEL CONSOLIDATION_MODEL OLLAMA_EMBEDDING_MODEL; do
        _val="${!_var:-}"
        if [[ -n "$_val" && -z "${_SEEN[$_val]:-}" ]]; then
            _SEEN[$_val]=1
            MODELS_TO_PULL+=("$_val")
        fi
    done

    for model in "${MODELS_TO_PULL[@]}"; do
        if docker exec janus-ollama ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qxF "$model"; then
            ok "$model already present"
        else
            log "Pulling ${BOLD}${model}${NC} — this may take several minutes on first run..."
            docker exec janus-ollama ollama pull "$model"
            ok "$model downloaded"
        fi
    done
fi

# ── Launch ────────────────────────────────────────────────────────────────────
hdr "Janus Process"
if [[ "$MODE" == "--cli" ]]; then
    log "Starting interactive CLI  (Ctrl-C to quit)..."
    exec python main.py
else
    log "Starting API server"
    log "  → http://localhost:8080        (REST)"
    log "  → http://localhost:8080/docs   (Swagger UI)"
    log "  → http://localhost:8080/redoc  (ReDoc)"
    exec uvicorn api.server:app --host 0.0.0.0 --port 8080 --reload
fi
