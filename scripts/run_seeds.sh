#!/usr/bin/env bash
# Run the main experiment for several seeds, then aggregate. Usage: bash scripts/run_seeds.sh [config] [seeds...]
set -euo pipefail
CFG=${1:-configs/default.yaml}; shift || true
SEEDS=${@:-1234 1235 1236}
export PYTHONPATH=src
BASE=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['out_dir'])")
for s in $SEEDS; do
  echo "=== seed $s ==="
  python -m swedit.run_experiment --config "$CFG" --seed "$s" --out-dir "$BASE/seed_$s"
done
python -m swedit.aggregate "$BASE"/seed_* | tee "$BASE/aggregate.txt"
bash scripts/freeze_env.sh "$BASE/environment.txt"
