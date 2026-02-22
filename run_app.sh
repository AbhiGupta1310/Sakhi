#!/bin/bash

# ==============================================================================
# Sakhi — Startup & Management Script
# ==============================================================================
# This script automates:
# 1. Killing stale processes on backend (8000) and frontend (5173) ports.
# 2. Starting the FastAPI backend with 'uv run'.
# 3. Waiting for the RAG pipeline to be healthy (embedding models load time).
# 4. Starting the React/Vite frontend.
# 5. Graceful shutdown of both on Ctrl+C.
# ==============================================================================

# Colors for logging
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

PROJECT_ROOT=$(pwd)

# ── Clean Architecture ────────────────────────────────────────────────────────

kill_port_process() {
    local port=$1
    local pid=$(lsof -t -i:$port)
    if [ -n "$pid" ]; then
        log_warn "Port $port is occupied by PID(s): $pid. Killing them..."
        kill -9 $pid 2>/dev/null
    fi
}

cleanup() {
    echo ""
    log_step "Stopping Sakhi..."
    [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    log_info "Peace out! 🙏✨"
    exit
}

# ── Phase 1: Cleanup ──────────────────────────────────────────────────────────

log_step "Phase 1: Port Cleanup"
kill_port_process 8000
kill_port_process 5173

# ── Phase 2: Start Backend ───────────────────────────────────────────────────

log_step "Phase 2: Starting Sakhi Backend"
log_info "Running: uv run python src/api/main.py"

# Start backend in background
uv run python src/api/main.py > backend.log 2>&1 &
BACKEND_PID=$!

log_info "Backend started (PID: $BACKEND_PID). Logging to backend.log"

# ── Phase 3: Wait for Health ──────────────────────────────────────────────────

log_step "Phase 3: Waiting for RAG Engine Readiness"
log_info "This involves loading BGE-M3 and ChromaDB. Please wait..."

MAX_RETRIES=45
RETRY_COUNT=0

while true; do
    # Check if process is still alive
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        log_error "Backend process died unexpectedly. See backend.log for details:"
        tail -n 10 backend.log
        exit 1
    fi

    # Check health endpoint
    HEALTH_RESP=$(curl -s http://localhost:8000/health 2>/dev/null)
    if echo "$HEALTH_RESP" | grep -q '"status":"healthy"'; then
        log_info "Backend is UP and HEALTHY! ✅"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        log_error "Backend failed to reach 'healthy' status within $MAX_RETRIES seconds."
        log_error "Last response: $HEALTH_RESP"
        cleanup
    fi

    # Visual progress
    printf "."
    sleep 1
done
echo ""

# ── Phase 4: Start Frontend ───────────────────────────────────────────────────

log_step "Phase 4: Starting Sakhi UI"
cd sakhi-ui || { log_error "sakhi-ui directory not found!"; cleanup; }

log_info "Running: npm run dev"
npm run dev &
FRONTEND_PID=$!

log_info "Sakhi UI started (PID: $FRONTEND_PID)"

# ── Phase 5: Monitoring ───────────────────────────────────────────────────────

log_step "Phase 5: Sakhi is Active! 🚀"
log_info "Frontend: http://localhost:5173"
log_info "Backend:  http://localhost:8000"
log_info "Press Ctrl+C to stop both servers."

# Trap interrupts to cleanup
trap cleanup INT TERM EXIT

# Keep script running to monitor logs or just wait
wait
