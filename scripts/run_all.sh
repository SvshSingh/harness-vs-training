#!/usr/bin/env bash
# Full experiment. Resumable: every step is skipped if its output already exists.
# Usage: bash scripts/run_all.sh            (env: MODEL, SEEDS, BUDGETS, BS)
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
SEEDS=${SEEDS:-"0 1 2"}
BUDGETS=${BUDGETS:-"250 1000"}
BS=${BS:-32}
PY=${PY:-python}

ev() {  # ev <out-prefix> <mode> [extra args]
  local out=$1 mode=$2; shift 2
  [[ -f "$out.summary.json" ]] || $PY -m hvt.evaluate --model "$MODEL" --mode "$mode" --batch-size "$BS" --out "$out" "$@"
}

mkdir -p results runs data

# 1. Base model, no weight changes.
ev results/base/plain plain
ev results/base/harness harness

# 2. Probe the base model and turn its errors into sampling weights.
ev results/probe/base_plain plain --split probe
[[ -f data/weights.json ]] || $PY -m hvt.targeting results/probe/base_plain.jsonl --out data/weights.json

# 3. Same budget, two data recipes, three seeds each.
for n in $BUDGETS; do
  for seed in $SEEDS; do
    for recipe in random targeted; do
      name=${recipe}_${n}_s${seed}
      train=data/$name.jsonl
      weights=(); [[ $recipe == targeted ]] && weights=(--weights data/weights.json)
      [[ -f $train ]] || $PY -m hvt.data --n "$n" --seed $((10000 + seed)) "${weights[@]}" --out "$train"
      [[ -f runs/$name/train_meta.json ]] || $PY -m hvt.train --model "$MODEL" --train "$train" --out "runs/$name" --seed "$seed"
      ev "results/$name/plain" plain --adapter "runs/$name"
      ev "results/$name/harness" harness --adapter "runs/$name"
    done
  done
done

$PY -m hvt.report
