#!/usr/bin/env bash
set -euo pipefail

trap 'kill 0' EXIT INT TERM
python3 -m uvicorn backend.main:app --reload --port 8000 &
npm run dev
