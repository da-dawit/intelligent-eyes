#!/usr/bin/env bash
# Encode and evaluate on a rented GPU, keeping the GPU busy (run from the repo root):
#   1. per dataset, ONE pass over the frames: each frame is decoded once and fed to all
#      encoders, plus the 2x high-res pass (baseline B1)
#   2. as soon as a dataset is encoded, its evaluations start on the CPU (one process per
#      encoder and seed) while the GPU moves on to the next dataset
#   3. Table I per dataset
# Expects $EYES_DATA/raw/{iwildcam_v2.0,ddr}, $EYES_DATA/processed/<dataset>/frames.csv and
# the encoder weights in the Hugging Face cache (DINOv3 is gated: copy its weights in, so
# the rented machine needs no access token). GPU use is logged to $EYES_DATA/logs/gpu.log.
set -uo pipefail
export EYES_DATA=${EYES_DATA:-/workspace/data} PYTHONPATH=$(pwd) HF_HUB_OFFLINE=1
PY=${PY:-python}
D=$EYES_DATA
ENCODERS=${ENCODERS:-"dinov2 clip siglip2 dinov3"}
DATASETS=${DATASETS:-"iwildcam ddr"}
BATCH=${BATCH:-128}
WORKERS=${WORKERS:-32}
EVAL_THREADS=${EVAL_THREADS:-8}
mkdir -p "$D/logs"
step() { echo "[$(date +%H:%M:%S)] $*"; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 10 > "$D/logs/gpu.log" &
smi=$!

evals=()
for ds in $DATASETS; do
    [ "$ds" = iwildcam ] && img="$D/raw/iwildcam_v2.0" || img="$D/raw/ddr"
    if [ -n "${WAIT_FOR:-}" ]; then        # e.g. WAIT_FOR="iwildcam:/workspace/extract.done"
        for w in $WAIT_FOR; do
            [ "${w%%:*}" = "$ds" ] && until [ -e "${w#*:}" ]; do sleep 15; done
        done
    fi
    step "encode $ds with $ENCODERS + hires (batch $BATCH, $WORKERS workers)"
    $PY scripts/cache_features.py --frames "$D/processed/$ds/frames.csv" --images "$img" --encoders $ENCODERS \
        --hires --out-root "$D/features/$ds" --batch "$BATCH" --workers "$WORKERS" > "$D/logs/cache_$ds.log" 2>&1 \
        || { step "FAILED encode $ds"; tail -20 "$D/logs/cache_$ds.log"; continue; }
    tail -1 "$D/logs/cache_$ds.log"
    for e in $ENCODERS; do
        for s in 0 1 2; do
            OMP_NUM_THREADS=$EVAL_THREADS MKL_NUM_THREADS=$EVAL_THREADS \
                $PY scripts/run_eval.py --dataset "$ds" --encoder "$e" --seeds $s > "$D/logs/eval_${ds}_${e}_seed$s.log" 2>&1 &
            evals+=($!)
        done
    done
    step "started evaluations for $ds"
done
kill $smi 2>/dev/null

step "encoding done; waiting for ${#evals[@]} evaluations"
fails=0
for p in "${evals[@]}"; do wait "$p" || fails=$((fails + 1)); done
step "evaluations finished, $fails failed"
for ds in $DATASETS; do $PY scripts/summarize.py --dataset "$ds" > "$D/logs/summary_$ds.log" 2>&1; done
step "ALL DONE"
