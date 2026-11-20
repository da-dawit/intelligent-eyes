#!/usr/bin/env bash
# The whole pipeline on a rented GPU (run from the repo root, data under $EYES_DATA):
#   1. fetch iWildCam and the DDR grading split from Hugging Face (pinned revisions)
#   2. encode every frame with each encoder, plus the 2x high-res pass (baseline B1)
#   3. evaluate each (dataset, encoder) on 3 seeds as soon as its features exist
#   4. Table I per dataset
# frames.csv for each dataset must already be in $EYES_DATA/processed/<dataset>/.
# DINOv3 is gated: its weights are expected in the local Hugging Face cache (copied in),
# so no access token is needed on the rented machine.
set -uo pipefail
export EYES_DATA=${EYES_DATA:-/workspace/data} PYTHONPATH=$(pwd)
D=$EYES_DATA
ENCODERS=${ENCODERS:-"dinov2 clip siglip2 dinov3"}
DATASETS=${DATASETS:-"iwildcam ddr"}
NC=$(nproc)
WORKERS=$(( NC > 16 ? 16 : NC ))
export OMP_NUM_THREADS=$(( NC / 8 > 1 ? NC / 8 : 1 )) MKL_NUM_THREADS=$(( NC / 8 > 1 ? NC / 8 : 1 ))
mkdir -p "$D/raw" "$D/logs"
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "fetching data"
python - <<'EOF'
import concurrent.futures as cf, os
from huggingface_hub import hf_hub_download, snapshot_download
raw = os.path.join(os.environ["EYES_DATA"], "raw")
def wilds():
    if os.path.isdir(os.path.join(raw, "iwildcam_v2.0", "train")):
        return "iWildCam already extracted"
    return hf_hub_download("pangwei/wilds-iwildcam", "iwildcam_v2.0.tar.gz", repo_type="dataset",
                           revision="78b5e7d858adb01cb5661c7b5f8666a4e91be2cb", local_dir=raw)
def ddr():
    return snapshot_download("YBoWen/DDR-dataset", repo_type="dataset", revision="cc2af821dbde42c545fd854d8e675c3bb74ca265",
                             allow_patterns=["DR_grading/*"], local_dir=os.path.join(raw, "ddr"), max_workers=32)
with cf.ThreadPoolExecutor(2) as ex:
    for f in [ex.submit(wilds), ex.submit(ddr)]:
        print(f.result())
EOF
if [ -f "$D/raw/iwildcam_v2.0.tar.gz" ]; then
    echo "05c872f46eee44315f79bc90f4098c69ae3799a2b95116db02292ea1b4e370d2  $D/raw/iwildcam_v2.0.tar.gz" | sha256sum -c - || exit 1
    mkdir -p "$D/raw/iwildcam_v2.0"                 # the archive has no top-level folder
    tar xzf "$D/raw/iwildcam_v2.0.tar.gz" -C "$D/raw/iwildcam_v2.0" && rm "$D/raw/iwildcam_v2.0.tar.gz"
fi
step "data ready: $(ls "$D/raw/iwildcam_v2.0/train" | wc -l) iWildCam images, $(find "$D/raw/ddr/DR_grading" -name '*.jpg' | wc -l) DDR images"

evals=()
for ds in $DATASETS; do
    [ "$ds" = iwildcam ] && img="$D/raw/iwildcam_v2.0" || img="$D/raw/ddr"
    for e in $ENCODERS; do
        offline=""; [ "$e" = dinov3 ] && offline="HF_HUB_OFFLINE=1"
        ok=1
        for mode in "" "--hires"; do
            tag="${ds}_${e}${mode:+_hr}"
            step "encode $tag"
            env $offline python scripts/cache_features.py --frames "$D/processed/$ds/frames.csv" --images "$img" \
                --encoder "$e" --out "$D/features/$ds/$e${mode:+_hr}" --batch 64 --workers "$WORKERS" $mode \
                > "$D/logs/cache_$tag.log" 2>&1 || { ok=0; step "FAILED $tag"; tail -5 "$D/logs/cache_$tag.log"; }
            grep -E "chunk .*frames/s" "$D/logs/cache_$tag.log" | tail -1
        done
        if [ $ok = 1 ]; then
            step "evaluate ${ds}_$e (background)"
            python scripts/run_eval.py --dataset "$ds" --encoder "$e" --seeds 0 1 2 > "$D/logs/eval_${ds}_$e.log" 2>&1 &
            evals+=($!)
        fi
    done
done

step "encoding done; waiting for ${#evals[@]} evaluations"
for p in "${evals[@]}"; do wait "$p" || step "an evaluation failed (pid $p)"; done
for ds in $DATASETS; do python scripts/summarize.py --dataset "$ds" > "$D/logs/summary_$ds.log" 2>&1; done
step "ALL DONE"
