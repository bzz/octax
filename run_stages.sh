#!/bin/sh
# Explicitly invoke a stage; no unattended work starts just by creating this file.
set -eu
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib}"
PY="$PWD/.venv/bin/python"
case "${1:-help}" in
  sanity) "$PY" sanity.py ;;
  smoke) "$PY" train.py ;;
  pilot)
    # 499,712 is the nearest complete 512-step rollout below 500K.
    "$PY" train.py total_timesteps=499712 eval_freq=65536 run_dir=runs/tetris/v2_pilot num_seeds=3
    for seed in 0 1 2; do
      "$PY" evaluate_checkpoint.py "runs/tetris/v2_pilot/seed$seed/checkpoint.msgpack" --out "runs/tetris/v2_pilot/seed$seed/heldout"
    done ;;
  transfer)
    source_checkpoint="${2:?Pass a Tetris checkpoint as the second argument}"
    # Match target seeds, budgets, optimizer settings and evaluation seeds.
    for game in brix pong; do
      for seed in 0 1 2; do
        for condition in scratch pretrained; do
          encoder=null
          if [ "$condition" = pretrained ]; then encoder="$source_checkpoint"; fi
          run="runs/$game/$condition/seed$seed"
          "$PY" train.py "env=$game" "seed=$seed" total_timesteps=499712 eval_freq=65536 "run_dir=$run" "encoder_checkpoint=$encoder"
          "$PY" evaluate_checkpoint.py "$run/checkpoint.msgpack" --out "$run/heldout"
        done
      done
    done ;;
  *) echo 'Usage: ./run_stages.sh sanity|smoke|pilot|transfer [Tetris checkpoint]' ;;
esac
