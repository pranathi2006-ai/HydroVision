#!/usr/bin/env bash
set -euo pipefail

trap 'kill 0' EXIT INT TERM
python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001 &
npm run dev
