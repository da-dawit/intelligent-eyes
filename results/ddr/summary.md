# Results: ddr

## clip  (3759 test clips, 3 seeds, rho=0.00, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -0.91 pts [-1.83, -0.02], McNemar p<=0.307 | >= +2 pts | fail |
| 2. Beats random | -0.95 pts [-1.40, -0.50] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -0.99 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.933 (pooled 0.933); adaptive eyes 0.885; greedy finds the best pair in 38% of clips, falls below 3/4 in 3.9% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.07, MC 0.04, closed~MC 0.16 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.110 vs no-R 0.317; glance 0.058 vs 0.058; All-9 0.126 vs 0.565 | <= 0.05 | fail |

Accuracy: glance 61.56, eyes(2) 60.65, random(2) 61.60, best pair (oracle) 71.84, All-9 60.57

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 60.99 | 4.0 | -0.35 [-1.24, +0.56] |
| B3 attention | 61.44 | 3.0 | -0.80 [-1.45, -0.13] |
| B4 motion | 61.75 | 3.0 | -1.10 [-1.81, -0.36] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 10.00 | 60.79, 60.89, 60.02 |
| all9_noR | 10.00 | 38.28, 39.13, 34.45 |
| attention_b2 | 3.00 | 61.43, 62.09, 60.81 |
| best_pair | 3.00 | 71.14, 72.44, 71.93 |
| eyes_b1 | 2.00 | 61.56, 61.98, 60.71 |
| eyes_b2 | 3.00 | 60.55, 61.27, 60.12 |
| eyes_b3 | 4.00 | 60.34, 60.95, 60.18 |
| eyes_b4 | 5.00 | 60.47, 61.05, 59.62 |
| eyes_mc_b2 | 3.00 | 60.65, 61.56, 59.91 |
| eyes_noR_b2 | 3.00 | 49.80, 49.99, 46.53 |
| eyes_overlapR_b2 | 3.00 | 54.99, 56.40, 54.83 |
| eyes_stop | 1.67 | 60.47, 62.33, 61.48 |
| glance | 1.00 | 61.56, 61.85, 61.27 |
| glance_noR | 1.00 | 61.56, 61.85, 61.27 |
| hires | 4.00 | 61.43, 60.89, 60.65 |
| motion_b2 | 3.00 | 61.67, 62.36, 61.21 |
| random_b1 | 2.00 | 61.79, 62.28, 61.65 |
| random_b2 | 3.00 | 61.55, 61.88, 61.36 |
| random_b3 | 4.00 | 61.52, 61.67, 61.13 |
| random_b4 | 5.00 | 61.29, 61.42, 60.77 |

Official metrics (mean over seeds):

| method | macro-F1 | quadratic kappa |
|---|---|---|
| glance | 0.369 | 0.589 |
| hires | 0.356 | 0.585 |
| random_b2 | 0.389 | 0.631 |
| motion_b2 | 0.387 | 0.630 |
| attention_b2 | 0.389 | 0.629 |
| eyes_b2 | 0.384 | 0.637 |
| eyes_stop | 0.379 | 0.624 |
| best_pair | 0.537 | 0.765 |
| all9 | 0.387 | 0.644 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | -0.91 | 3759 |

## dinov2  (3759 test clips, 3 seeds, rho=0.00, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -0.89 pts [-1.83, +0.04], McNemar p<=0.209 | >= +2 pts | fail |
| 2. Beats random | -0.82 pts [-1.32, -0.33] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -2.47 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.941 (pooled 0.944); adaptive eyes 0.887; greedy finds the best pair in 50% of clips, falls below 3/4 in 6.3% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.10, MC 0.05, closed~MC 0.26 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.074 vs no-R 0.312; glance 0.054 vs 0.054; All-9 0.097 vs 0.581 | <= 0.05 | fail |

Accuracy: glance 64.08, eyes(2) 63.19, random(2) 64.01, best pair (oracle) 75.50, All-9 61.61

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 66.29 | 4.0 | -3.10 [-4.12, -2.09] |
| B3 attention | 64.80 | 3.0 | -1.61 [-2.38, -0.87] |
| B4 motion | 64.84 | 3.0 | -1.65 [-2.44, -0.85] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 10.00 | 61.05, 62.38, 61.40 |
| all9_noR | 10.00 | 32.32, 35.65, 36.37 |
| attention_b2 | 3.00 | 64.59, 65.34, 64.49 |
| best_pair | 3.00 | 75.31, 75.76, 75.42 |
| eyes_b1 | 2.00 | 63.85, 64.22, 64.25 |
| eyes_b2 | 3.00 | 63.10, 63.13, 63.34 |
| eyes_b3 | 4.00 | 62.49, 62.76, 62.28 |
| eyes_b4 | 5.00 | 61.80, 62.38, 62.09 |
| eyes_mc_b2 | 3.00 | 62.41, 62.89, 63.26 |
| eyes_noR_b2 | 3.00 | 46.77, 48.87, 48.79 |
| eyes_overlapR_b2 | 3.00 | 54.88, 56.45, 56.29 |
| eyes_stop | 1.35 | 64.11, 64.64, 64.51 |
| glance | 1.00 | 63.85, 64.06, 64.33 |
| glance_noR | 1.00 | 63.85, 64.06, 64.33 |
| hires | 4.00 | 66.69, 65.92, 66.27 |
| motion_b2 | 3.00 | 64.54, 64.99, 64.99 |
| random_b1 | 2.00 | 64.11, 64.33, 64.35 |
| random_b2 | 3.00 | 63.71, 64.26, 64.06 |
| random_b3 | 4.00 | 63.46, 64.12, 63.82 |
| random_b4 | 5.00 | 63.19, 63.71, 63.52 |

Official metrics (mean over seeds):

| method | macro-F1 | quadratic kappa |
|---|---|---|
| glance | 0.402 | 0.632 |
| hires | 0.409 | 0.668 |
| random_b2 | 0.422 | 0.667 |
| motion_b2 | 0.423 | 0.656 |
| attention_b2 | 0.433 | 0.671 |
| eyes_b2 | 0.426 | 0.670 |
| eyes_stop | 0.411 | 0.666 |
| best_pair | 0.589 | 0.799 |
| all9 | 0.417 | 0.661 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | -0.89 | 3759 |

## dinov3  (3759 test clips, 3 seeds, rho=0.00, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -1.45 pts [-2.30, -0.57], McNemar p<=0.0648 | >= +2 pts | fail |
| 2. Beats random | -0.99 pts [-1.43, -0.54] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -3.76 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.930 (pooled 0.933); adaptive eyes 0.881; greedy finds the best pair in 50% of clips, falls below 3/4 in 9.9% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.07, MC 0.03, closed~MC 0.20 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.087 vs no-R 0.315; glance 0.059 vs 0.059; All-9 0.099 vs 0.551 | <= 0.05 | fail |

Accuracy: glance 64.68, eyes(2) 63.23, random(2) 64.22, best pair (oracle) 74.55, All-9 60.92

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 67.06 | 4.0 | -3.82 [-4.69, -2.95] |
| B3 attention | 64.31 | 3.0 | -1.07 [-1.82, -0.36] |
| B4 motion | 64.31 | 3.0 | -1.07 [-1.82, -0.36] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 10.00 | 60.68, 60.97, 61.11 |
| all9_noR | 10.00 | 35.75, 38.44, 40.06 |
| attention_b2 | 3.00 | 64.06, 64.51, 64.35 |
| best_pair | 3.00 | 74.33, 74.67, 74.65 |
| eyes_b1 | 2.00 | 63.98, 64.03, 64.41 |
| eyes_b2 | 3.00 | 62.94, 63.47, 63.29 |
| eyes_b3 | 4.00 | 62.22, 63.08, 63.18 |
| eyes_b4 | 5.00 | 61.48, 62.62, 63.00 |
| eyes_mc_b2 | 3.00 | 63.10, 63.47, 63.53 |
| eyes_noR_b2 | 3.00 | 48.66, 50.65, 51.56 |
| eyes_overlapR_b2 | 3.00 | 56.05, 56.29, 57.97 |
| eyes_stop | 1.34 | 64.54, 64.38, 64.25 |
| glance | 1.00 | 64.62, 64.49, 64.94 |
| glance_noR | 1.00 | 64.62, 64.49, 64.94 |
| hires | 4.00 | 66.91, 66.88, 67.38 |
| motion_b2 | 3.00 | 64.06, 64.51, 64.35 |
| random_b1 | 2.00 | 64.30, 64.17, 64.68 |
| random_b2 | 3.00 | 64.05, 64.25, 64.37 |
| random_b3 | 4.00 | 63.86, 63.98, 63.89 |
| random_b4 | 5.00 | 63.24, 63.58, 63.51 |

Official metrics (mean over seeds):

| method | macro-F1 | quadratic kappa |
|---|---|---|
| glance | 0.410 | 0.643 |
| hires | 0.423 | 0.675 |
| random_b2 | 0.424 | 0.670 |
| motion_b2 | 0.420 | 0.655 |
| attention_b2 | 0.420 | 0.655 |
| eyes_b2 | 0.418 | 0.672 |
| eyes_stop | 0.410 | 0.660 |
| best_pair | 0.573 | 0.791 |
| all9 | 0.404 | 0.667 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | -1.45 | 3759 |

## siglip2  (3759 test clips, 3 seeds, rho=0.00, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -1.31 pts [-2.24, -0.37], McNemar p<=0.109 | >= +2 pts | fail |
| 2. Beats random | -1.16 pts [-1.64, -0.69] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -1.85 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.876 (pooled 0.877); adaptive eyes 0.832; greedy finds the best pair in 37% of clips, falls below 3/4 in 18.9% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.10, MC 0.07, closed~MC 0.26 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.120 vs no-R 0.312; glance 0.061 vs 0.061; All-9 0.150 vs 0.540 | <= 0.05 | fail |

Accuracy: glance 61.78, eyes(2) 60.47, random(2) 61.63, best pair (oracle) 72.28, All-9 59.93

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 61.06 | 4.0 | -0.59 [-1.46, +0.25] |
| B4 motion | 62.52 | 3.0 | -2.05 [-2.84, -1.28] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 10.00 | 60.02, 59.99, 59.78 |
| all9_noR | 10.00 | 40.65, 41.74, 37.62 |
| best_pair | 3.00 | 72.25, 72.81, 71.77 |
| eyes_b1 | 2.00 | 61.27, 61.80, 61.24 |
| eyes_b2 | 3.00 | 60.63, 60.89, 59.88 |
| eyes_b3 | 4.00 | 60.12, 60.44, 59.99 |
| eyes_b4 | 5.00 | 60.34, 60.34, 60.28 |
| eyes_mc_b2 | 3.00 | 60.63, 60.23, 60.20 |
| eyes_noR_b2 | 3.00 | 49.14, 50.09, 47.86 |
| eyes_overlapR_b2 | 3.00 | 55.28, 56.61, 55.81 |
| eyes_stop | 1.97 | 60.49, 61.05, 61.00 |
| glance | 1.00 | 61.53, 61.98, 61.82 |
| glance_noR | 1.00 | 61.53, 61.98, 61.82 |
| hires | 4.00 | 61.08, 60.89, 61.21 |
| motion_b2 | 3.00 | 62.49, 62.73, 62.33 |
| random_b1 | 2.00 | 61.80, 62.34, 61.77 |
| random_b2 | 3.00 | 61.55, 61.90, 61.43 |
| random_b3 | 4.00 | 61.29, 61.70, 61.21 |
| random_b4 | 5.00 | 61.03, 61.52, 61.06 |

Official metrics (mean over seeds):

| method | macro-F1 | quadratic kappa |
|---|---|---|
| glance | 0.368 | 0.598 |
| hires | 0.345 | 0.587 |
| random_b2 | 0.391 | 0.649 |
| motion_b2 | 0.395 | 0.650 |
| eyes_b2 | 0.392 | 0.646 |
| eyes_stop | 0.389 | 0.645 |
| best_pair | 0.548 | 0.782 |
| all9 | 0.386 | 0.661 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | -1.31 | 3759 |
