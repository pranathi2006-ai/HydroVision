#!/usr/bin/env bash
set -euo pipefail

hydrovision_backend_pid=""

cleanup_hydrovision() {
  if [[ -n "$hydrovision_backend_pid" ]] && kill -0 "$hydrovision_backend_pid" 2>/dev/null; then
    kill "$hydrovision_backend_pid" 2>/dev/null || true
    wait "$hydrovision_backend_pid" 2>/dev/null || true
  fi
}

trap cleanup_hydrovision EXIT INT TERM

python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 &
hydrovision_backend_pid=$!

# The findings snapshot is fetched as soon as the page opens. Wait for FastAPI
# so the first request cannot lose a startup race and leave the UI empty.
for _ in {1..60}; do
  if curl --fail --silent http://127.0.0.1:8001/api/health >/dev/null; then
    npm run dev:web
    exit $?
  fi
  if ! kill -0 "$hydrovision_backend_pid" 2>/dev/null; then
    wait "$hydrovision_backend_pid"
    exit $?
  fi
  sleep 0.5
done

echo "HydroVision API did not become ready on port 8001." >&2
exit 1
