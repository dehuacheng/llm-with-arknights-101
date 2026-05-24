#!/bin/bash
# Stage 02 sweep driver — runs all nine configs in 02_pretrain/configs/
# sequentially, logging to data/sweep_logs/. The configs are the source of
# reproducibility; this script is convenience over typing nine `python ...`
# lines. Output paths land under data/ (git-ignored).
set -u

# Run from repo root regardless of cwd. The script lives at
# 02_pretrain/scripts/run_sweep.sh; ../.. is the repo root.
cd "$(dirname "$0")/../.."

mkdir -p data/sweep_logs
rm -f data/sweep_logs/status data/sweep_logs/progress

for cfg in tiny_32k small_32k large_32k small_8k small_16k ctx_256 ctx_1024 ctx_2048 ctx_4096; do
    echo "[$(date +%H:%M:%S)] start $cfg" >> data/sweep_logs/progress
    .venv/bin/python -u 02_pretrain/train.py \
        --config 02_pretrain/configs/$cfg.yaml \
        > data/sweep_logs/$cfg.log 2>&1
    echo "[$(date +%H:%M:%S)] done $cfg (exit $?)" >> data/sweep_logs/progress
done
echo done > data/sweep_logs/status
