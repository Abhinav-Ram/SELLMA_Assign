#!/usr/bin/env bash
# Record the exact environment for reproducibility. Usage: bash scripts/freeze_env.sh [outfile]
OUT=${1:-results/environment.txt}
mkdir -p "$(dirname "$OUT")"
{
  echo "date: $(date -u +%FT%TZ)"
  echo "python: $(python --version 2>&1)"
  echo "platform: $(python -c 'import platform;print(platform.platform())')"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
  echo "--- pip freeze ---"
  python -m pip freeze
} > "$OUT"
echo "wrote $OUT"
