# Intelligent Eyes

**Teaching frozen ViT encoders where — and when — to look in video.**
DA411 Computational Mathematics, final project (Option A). Dawit Chun.

Frozen encoders such as CLIP and DINOv2 see each frame shrunk to 224 px and miss small things.
The eyes glance at every frame of a clip, then spend a small budget of 2× zoomed *looks* on the
(frame, crop) pairs that Bayes' rule and information theory predict will teach the model the
most. Nothing is trained except linear classifiers and a ridge regression; the encoders stay frozen.

- Proposal: [docs/proposal.pdf](docs/proposal.pdf)
- Every equation, derived step by step: [docs/derivations.pdf](docs/derivations.pdf)

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

Each numbered result has a test in [tests/test_derivations.py](tests/test_derivations.py) that
checks the code against it.

## One change from the proposal, found before any real data

On synthetic clips with a *known* R ([eyes/synthetic.py](eyes/synthetic.py)), combining every
crop with eq. (1) made the model *worse* than ignoring correlation. The cause: the four corner
crops tile the frame, so the geometric overlap table `A` is singular and `w = A⁻¹1` gives the
centre crop weight 0. Prop. 3.1's model ("every vote is a noisy copy of one L") fits repeats in
*time*, not crops that see *different* parts of a frame. The fix is to shrink the within-frame table,
`A_λ = (1-λ)A + λI`, with λ chosen by held-out log-likelihood ([eyes/correlation.py: shrink](eyes/correlation.py),
[eyes/fit.py](eyes/fit.py)). Time (ρ) is not shrunk.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest          # 33 tests: derivations + controller on synthetic clips
```

Data lives outside the repo, in `../data` by default (set `EYES_DATA` to move it).
Hardware used: Apple M5 Pro (MPS), 51 GB RAM.

## Pipeline

```bash
export PYTHONPATH=.
# 0. data: WILDS iWildCam v2.0 (12 GB, frames 448 px tall). The downloader is parallel,
#    resumable across dropped connections, and checks the SHA-256.
python scripts/download.py \
    https://huggingface.co/datasets/pangwei/wilds-iwildcam/resolve/78b5e7d858adb01cb5661c7b5f8666a4e91be2cb/iwildcam_v2.0.tar.gz \
    $EYES_DATA/raw/iwildcam_v2.0.tar.gz --sha256 05c872f46eee44315f79bc90f4098c69ae3799a2b95116db02292ea1b4e370d2
tar xzf $EYES_DATA/raw/iwildcam_v2.0.tar.gz -C $EYES_DATA/raw

# 1. clips: WILDS sequences -> bursts of 1-10 frames, one species each
python scripts/build_clips.py --dataset iwildcam --root $EYES_DATA/raw/iwildcam_v2.0 \
    --out $EYES_DATA/processed/iwildcam

# 2. encode every frame once: glance, 9 crops, crop previews (resumable)
python scripts/cache_features.py --frames $EYES_DATA/processed/iwildcam/frames.csv \
    --images $EYES_DATA/raw/iwildcam_v2.0 --encoder dinov2 --out $EYES_DATA/features/iwildcam/dinov2

# 3. fit on 10% held-out training cameras, run every method on the test clips, 3 seeds
python scripts/run_eval.py --dataset iwildcam --encoder dinov2 --seeds 0 1 2

# 4. Table I of the proposal + accuracy-vs-passes figure
python scripts/summarize.py --dataset iwildcam
```

Encoders: `clip` (CLIP B/16), `siglip2` (SigLIP 2 B/16), `dinov2` (DINOv2 B/14), `dinov3`
(DINOv3 B/16 — gated on Hugging Face: accept the licence, then `hf auth login`).

## Methods compared

| name | what it does | encoder passes |
|---|---|---|
| `glance` (B0) | every frame once, combined by eq. (1) | T |
| `random_b2` (B2) | + 2 random looks (10 draws) | T + 2 |
| `motion_b2` (B4) | + 2 looks where consecutive frames differ most | T + 2 |
| `eyes_b2` | + 2 looks chosen by closed-form EIG | T + 2 |
| `eyes_mc_b2` | same, with Monte Carlo EIG | T + 2 |
| `eyes_noR_b2` | eyes treating every observation as independent | T + 2 |
| `eyes_stop` | eyes with up to 4 looks and the ε stopping rule | ≤ T + 4 |
| `best_pair` | the pair that puts most belief on the true species (oracle reference) | T + 2 |
| `all9` | every crop of every frame (reference) | 10T |

## Layout

```
eyes/         the method (geometry, correlation, belief, predictor, eig, controller, pairs, fit, metrics)
              and the data side (data, views, encoders, classifier, cache, paths)
scripts/      build_clips -> cache_features -> run_eval -> summarize; benchmark_encoder
tests/        derivation checks, pair formulas, synthetic end-to-end checks
docs/         proposal and derivations
```

## Data and model licences

iWildCam (WILDS v2.0, CDLA-Permissive), DDR (CC BY-NC-SA 4.0, non-commercial). Encoders are
used frozen under their own licences (DINOv3 requires accepting Meta's licence).
