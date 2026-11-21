# Intelligent Eyes

**Teaching frozen ViT encoders where — and when — to look in video.**
DA411 Computational Mathematics, final project (Option A). Dawit Chun.

Frozen encoders such as CLIP and DINOv2 see each frame shrunk to 224 px and miss small things.
The eyes glance at every frame of a clip, then spend a small budget of 2× zoomed *looks* on the
(frame, crop) pairs that Bayes' rule and information theory predict will teach the model the
most. Nothing is trained except linear classifiers and a ridge regression; the encoders stay frozen.

- Proposal: [docs/proposal.pdf](docs/proposal.pdf)
- Every equation, one small step at a time, matched to the code (including the changes made after the first
  results): [docs/derivations_step_by_step.pdf](docs/derivations_step_by_step.pdf). The original, denser
  derivations from the proposal: [docs/derivations.pdf](docs/derivations.pdf)
- Project page with the method in brief, 19 videos (where each method looks, the four encoders on one clip, a
  stopping-rule gallery with failures, before and after the fix) and the benchmarks (success metrics, one change at
  a time, synthetic clips, DDR); open in a browser, works offline:
  [docs/project_page/index.html](docs/project_page/index.html), rebuilt by `scripts/make_videos.py` and
  `scripts/make_project_page.py`. Online: <https://da-dawit.github.io/intelligent-eyes/> (GitHub Pages serves
  `docs/` from `main`; `docs/index.html` forwards to the page).

## How the code maps to the mathematics

| Derivation | What it says | Code |
|---|---|---|
| Prop. 3.1, eq. (1) | combine correlated votes with weights `w = R⁻¹1` | `eyes/belief.py: posterior`, `eyes/correlation.py: evidence_weights` |
| Prop. 3.2 | feeding an observation twice cannot change the answer | `tests/test_derivations.py::test_repeat_cannot_change_the_answer` |
| Prop. 3.3 | T still frames count as `(T-(T-2)ρ)/(1+ρ)` | `eyes/correlation.py: ar1_effective_frames` |
| Sec. 3.1 | `R = T_ρ ⊗ A`, fitting `A` and `ρ` from residuals | `eyes/correlation.py: clip_correlation, fit_A, fit_rho` |
| Prop. 4.1 | adding one look: block inverse, innovation, `c_j` | `eyes/belief.py: add_one_look_terms` |
| Prop. 5.1, eq. (2) | ridge prediction of a look, then Gaussian conditioning | `eyes/predictor.py: fit_ridge, condition_on_looks` |
| Thm. 6.5, eq. (3) | `EIG ≈ ½(mᵀFm + v tr F)` | `eyes/eig.py: eig_closed_form` (and `eig_monte_carlo`) |
| Sec. 8, metric 4 | greedy vs. the best of all pairs: Prop. 4.1 with a 2×2 block, eq. (3) for the pair | `eyes/pairs.py: all_pairs, pair_report` |
| Cor. 7.1 | stopping rule: EIG → 0 as the belief sharpens | `eyes/controller.py: run(eps=...)`, `eyes/fit.py: choose_eps` |
| Sec. 10 | ECE, McNemar, camera bootstrap, Spearman | `eyes/metrics.py` |

| new: crop votes | a crop may show only background: `ℓ(y) = log(q(y∣c)/π(y) + q(empty∣c))` | `eyes/boxes.py: mixture_votes, crop_labels` |
| new: gated fusion | a look whose crop shows no animal stays out of eq. (1); `R` fitted on the rest | `eyes/controller.py: enters`; `eyes/correlation.py: fit_A(mask), fit_rho(mask)` |
| new: shared crop error | `R = T_ρ ⊗ A_λ + τ 1_c 1_cᵀ`: crop evidence levels off near `1/τ` | `eyes/correlation.py: clip_correlation`; `eyes/fit.py: choose_R` |
| new: two-branch EIG | `EIG = P(animal)·EIG(animal branch)` (the background branch adds nothing) | `eyes/controller.py: eig_mixture` |

Each numbered result has a test in [tests/test_derivations.py](tests/test_derivations.py) that
checks the code against it; the new pieces are checked in [tests/test_boxes.py](tests/test_boxes.py),
[tests/test_controller.py](tests/test_controller.py) and [tests/test_derivations.py](tests/test_derivations.py).

## Results (iWildCam, WILDS OOD test cameras, 3,930 clips, 3 seeds)

Clip accuracy (%), 2 looks. *First run*: the method exactly as proposed. *Final*: with the changes below.

| encoder | glance | eyes, first run | **eyes, final** | eyes + stopping (looks/clip) | motion (B4) | attention (B3) | high-res (B1, 4T) | All-9 (10T) |
|---|---|---|---|---|---|---|---|---|
| CLIP | 67.1 | 64.2 | **76.4** | 76.2 (1.6) | 75.8 | 70.8 | 65.9 | 75.1 |
| DINOv2 | 81.6 | 79.5 | **86.1** | 86.3 (0.9) | 85.5 | 85.1 | 85.4 | 86.0 |
| DINOv3 | 78.2 | 75.3 | **84.1** | 84.1 (1.0) | 83.0 | 84.7 | 82.2 | 84.2 |
| SigLIP 2 | 69.3 | 64.8 | **78.9** | 78.9 (1.2) | 78.2 | — | 68.8 | 77.6 |

With the final method all six success metrics of the proposal pass for CLIP, DINOv3 and SigLIP 2,
and five for DINOv2 (ranking 0.28 against 0.3): see [results/iwildcam/summary_final.md](results/iwildcam/summary_final.md).

## Changes from the proposal, and why

**1. Shrinking the within-frame table (found on synthetic data).** On synthetic clips with a *known* R ([eyes/synthetic.py](eyes/synthetic.py)), combining every
crop with eq. (1) made the model *worse* than ignoring correlation. The cause: the four corner
crops tile the frame, so the geometric overlap table `A` is singular and `w = A⁻¹1` gives the
centre crop weight 0. Prop. 3.1's model ("every vote is a noisy copy of one L") fits repeats in
*time*, not crops that see *different* parts of a frame. The fix is to shrink the within-frame table,
`A_λ = (1-λ)A + λI`, with λ chosen by held-out log-likelihood ([eyes/correlation.py: shrink](eyes/correlation.py),
[eyes/fit.py](eyes/fit.py)). Time (ρ) is not shrunk.

**2. Crop labels from animal boxes (the first real failure mode).** As proposed, every crop was
labelled with the clip's species. On real data every look *lowered* accuracy and All-9 lost 15–20
points. [scripts/diagnose_crops.py](scripts/diagnose_crops.py) measured why with MegaDetector v4
boxes (used for training labels and diagnosis only, never at test time): two thirds of crops show
no animal, so the crop classifier learned "this scenery means this species" and was right on
only 43% of the crops that *do* hold the animal. With an extra *empty* class — a crop gets the
species if it holds the animal, *empty* if it touches none — the vote follows from Bayes' rule
as `log(q(y∣c)/π(y) + q(empty∣c))` (derivation in [eyes/boxes.py](eyes/boxes.py)). Accuracy on
animal crops rose to 86%.

**3. Gated fusion.** Eq. (1) treats every vote as a noisy copy of the same evidence, so a
background crop's vote of 0 read as evidence *against* the glance's answer, and EIG expected
that spurious change. A look whose own crop classifier says *empty* (q ≥ ½) now costs its pass
but stays out of eq. (1), and `R` is fitted only on observations that enter it.

**4. A shared crop error in R.** With many looks, crop votes still over-counted: their errors are
largely shared within a clip (the same light, the same background). `τ 1_c 1_cᵀ` adds that shared
error; τ is fitted with λ by held-out log-likelihood (0.4–0.8: all the crops of a clip together
count about as much as one more frame). Extra looks now never hurt.

**5. Two-branch EIG.** A gated look either shows the animal or changes nothing, so
`EIG = P(animal) · EIG(animal branch)`, exact as an expectation, with eq. (3) for the branch.
`P(animal)` comes from a linear classifier on the glance's own tokens over the crop (AUC 0.97–0.98
on test cameras, better than pixel motion at 0.82 or CLS attention at 0.61–0.94).

**6. The look predictor starts from a guess.** Eq. (2) now also takes, as an input, the vote that
a linear classifier on the glance's own tokens over the crop guesses for that crop (the same species +
*empty* labels as change 2), so the ridge regression only has to correct that guess
([scripts/run_eval.py: predictor_inputs](scripts/run_eval.py)).

**7. Conditioning on looks only.** Prop. 5.1 conditions the predicted votes on the looks already
taken, not on the glances: the predictor already sees the glance, so conditioning on it again would
count it twice ([eyes/predictor.py: condition_on_looks](eyes/predictor.py)).

**A bug found on the way.** In float16 the eager attention pass turned every DINOv3 attention map
into NaN, which silently broke its B3 baseline; attention now runs in float32
([eyes/encoders.py](eyes/encoders.py), repaired caches with [scripts/fix_attention.py](scripts/fix_attention.py)).

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest          # 43 tests: derivations, new pieces, controller on synthetic clips
PYTHONPATH=. .venv/bin/python scripts/synthetic_study.py    # the method on synthetic clips
```

Dependencies (installed by the second line, from `pyproject.toml`): Python 3.11, PyTorch ≥ 2.5,
torchvision, Transformers ≥ 4.56, safetensors, NumPy, SciPy, pandas, scikit-learn, Pillow,
Matplotlib, tqdm; pytest for the tests. Optional: ffmpeg (project page videos); a CUDA GPU or
Apple MPS for encoding.

The tests and the synthetic study need no data. The summaries, fitted parameters, rankings and figures of every run are in
`results/`; the per-clip outputs behind them (`clips.csv.gz`, about 15 MB per run) are kept out of git
and are rebuilt by `run_eval.py`.

Data lives outside the repo, in `../data` by default (set `EYES_DATA` to move it).
Hardware used: development on an Apple M5 Pro (MPS); the full runs on one rented RTX PRO 6000
(Blackwell, 96 GB) with [scripts/cloud_run.sh](scripts/cloud_run.sh).

## Pipeline

```bash
export PYTHONPATH=.
# 0. data: WILDS iWildCam v2.0 (12 GB, frames 448 px tall). The downloader is parallel,
#    resumable across dropped connections, and checks the SHA-256.
python scripts/download.py \
    https://huggingface.co/datasets/pangwei/wilds-iwildcam/resolve/78b5e7d858adb01cb5661c7b5f8666a4e91be2cb/iwildcam_v2.0.tar.gz \
    $EYES_DATA/raw/iwildcam_v2.0.tar.gz --sha256 05c872f46eee44315f79bc90f4098c69ae3799a2b95116db02292ea1b4e370d2
mkdir -p $EYES_DATA/raw/iwildcam_v2.0    # the archive has no top-level folder
tar xzf $EYES_DATA/raw/iwildcam_v2.0.tar.gz -C $EYES_DATA/raw/iwildcam_v2.0

# 1. clips: WILDS sequences -> bursts of 1-10 frames, one species each
python scripts/build_clips.py --dataset iwildcam --root $EYES_DATA/raw/iwildcam_v2.0 \
    --out $EYES_DATA/processed/iwildcam

# 2. encode every frame once: glance, 9 crops, crop previews, CLS attention, and (--hires)
#    the whole frame at 2x for baseline B1. Each frame is decoded once for all encoders listed.
python scripts/cache_features.py --frames $EYES_DATA/processed/iwildcam/frames.csv \
    --images $EYES_DATA/raw/iwildcam_v2.0 --encoders dinov2 clip siglip2 dinov3 --hires \
    --out-root $EYES_DATA/features/iwildcam

# 3. animal boxes for the crop labels (82 MB, read out of the 169 GB LILA archive by range requests)
python scripts/fetch_boxes.py --out $EYES_DATA/raw/iwildcam_mdv4

# 4. fit on 10% held-out training cameras, run every method on the test clips, 3 seeds.
#    The final method; drop the flags for the method exactly as proposed.
python scripts/run_eval.py --dataset iwildcam --encoder dinov2 --seeds 0 1 2 \
    --crop-labels boxes --predictor preview --vote-shape none --fusion gated --crop-effect fit

# 5. Table I of the proposal + accuracy-vs-passes figure
python scripts/summarize.py --dataset iwildcam
```

DDR (one-frame clips, grades 0-4) runs the same steps with `--dataset ddr`. The grading split
(13,673 photos, the official train/valid/test lists) is fetched from a Hugging Face copy, pinned
to one revision; its file counts match the paper:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('YBoWen/DDR-dataset', \
    repo_type='dataset', revision='cc2af821dbde42c545fd854d8e675c3bb74ca265', \
    allow_patterns=['DR_grading/*'], local_dir='$EYES_DATA/raw/ddr')"
python scripts/build_clips.py --dataset ddr --root $EYES_DATA/raw/ddr --out $EYES_DATA/processed/ddr
```

Encoders: `clip` (CLIP B/16), `siglip2` (SigLIP 2 B/16), `dinov2` (DINOv2 B/14), `dinov3`
(DINOv3 B/16 — gated on Hugging Face: accept the licence, then `hf auth login`).

## Methods compared

| name | what it does | encoder passes |
|---|---|---|
| `glance` (B0) | every frame once, combined by eq. (1) | T |
| `hires` (B1) | every frame once at 2× resolution (448 px, 4× the tokens), combined by eq. (1) | 4T |
| `random_b2` (B2) | + 2 random looks (10 draws) | T + 2 |
| `attention_b2` (B3) | + 2 looks where the glance's CLS attention is highest | T + 2 |
| `motion_b2` (B4) | + 2 looks where consecutive frames differ most | T + 2 |
| `eyes_b2` | + 2 looks chosen by closed-form EIG | T + 2 |
| `eyes_mc_b2` | same, with Monte Carlo EIG | T + 2 |
| `eyes_noR_b2` | eyes treating every observation as independent | T + 2 |
| `eyes_stop` | eyes with up to 4 looks and the ε stopping rule | ≤ T + 4 |
| `eyesmix_b2` | + 2 looks chosen by the two-branch EIG (final method) | T + 2 |
| `eyesmix_stop` | two-branch EIG, up to 4 looks, ε stopping rule (final method) | ≤ T + 4 |
| `best_pair` | the pair that puts most belief on the true species (oracle reference) | T + 2 |
| `all9` | every crop of every frame (reference) | 10T |

## Layout

```
eyes/         the method (geometry, correlation, belief, predictor, eig, controller, pairs, boxes, fit, metrics)
              and the data side (data, views, encoders, classifier, cache, paths)
scripts/      build_clips -> cache_features -> fetch_boxes -> run_eval -> summarize;
              diagnose_crops (failure-mode analysis), cloud_run.sh, benchmark_encoder, fix_attention
tests/        derivation checks, pair formulas, synthetic end-to-end checks
docs/         proposal, derivations, and the project page (docs/project_page/)
```

## Data and model licences

iWildCam (WILDS v2.0, CDLA-Permissive; MegaDetector v4 boxes from the iWildCam 2022 metadata on
LILA, CDLA-Permissive), DDR (CC BY-NC-SA 4.0, non-commercial). Encoders are
used frozen under their own licences (DINOv3 requires accepting Meta's licence).
