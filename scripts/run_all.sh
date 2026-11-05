#!/usr/bin/env bash
# Full EmoDial pipeline. Usage: bash scripts/run_all.sh [configs/default.yaml]
set -euo pipefail
CFG=${1:-configs/default.yaml}
C="--config $CFG"
SEEDS=$(python -c "import yaml,sys;print(' '.join(map(str,yaml.safe_load(open('$CFG')).get('generator',{}).get('seeds',[42]))))" 2>/dev/null || echo 42)

python scripts/01_prepare_data.py $C
python scripts/02_train_recogniser.py $C
python scripts/03_cache_emotions.py $C          # check data/cache/delta_stats.json (12 Oct checkpoint)
python scripts/04_train_generator.py $C --stage base
for SEED in $SEEDS; do
  for V in b0 b1 etg_nodelta etg_nogate etg; do
    python scripts/04_train_generator.py $C --variant $V --seed $SEED
    python scripts/05_generate.py $C --variant $V --seed $SEED --swap
  done
done
python scripts/06_evaluate.py $C
python scripts/07_human_eval.py $C make
