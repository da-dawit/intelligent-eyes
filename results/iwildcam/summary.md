# Results: iwildcam

## clip  (3930 test clips, 3 seeds, rho=0.88, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -4.27 pts [-6.84, -1.58], McNemar p<=0.00126 | >= +2 pts | fail |
| 2. Beats random | +0.37 pts [-0.87, +1.36] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -16.56 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.980 (pooled 0.982); adaptive eyes 0.863; greedy finds the best pair in 50% of clips, falls below 3/4 in 0.2% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.11, MC 0.15, closed~MC 0.39 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.030 vs no-R 0.435; glance 0.052 vs 0.373; All-9 0.079 vs 0.805 | <= 0.05 | PASS |

Accuracy: glance 68.44, eyes(2) 64.16, random(2) 63.79, best pair (oracle) 87.48, All-9 51.87

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 65.89 | 27.8 | -1.72 [-4.24, +0.23] |
| B3 attention | 65.32 | 8.9 | -1.15 [-4.17, +2.02] |
| B4 motion | 74.00 | 8.9 | -9.84 [-14.31, -5.97] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 58.37, 48.04, 49.22 |
| all9_noR | 69.48 | 16.51, 11.09, 11.50 |
| attention_b2 | 8.95 | 66.06, 64.17, 65.74 |
| best_pair | 8.95 | 87.63, 87.12, 87.71 |
| eyes_b1 | 7.95 | 64.53, 63.10, 69.23 |
| eyes_b2 | 8.95 | 65.22, 60.56, 66.73 |
| eyes_b3 | 9.95 | 63.66, 56.36, 63.22 |
| eyes_b4 | 10.95 | 61.48, 54.50, 58.08 |
| eyes_mc_b2 | 8.95 | 64.27, 59.26, 63.55 |
| eyes_noR_b2 | 8.95 | 38.65, 34.96, 37.52 |
| eyes_overlapR_b2 | 8.95 | 66.08, 60.89, 67.09 |
| eyes_stop | 8.96 | 63.26, 57.94, 63.30 |
| glance | 6.95 | 68.32, 68.24, 68.77 |
| glance_noR | 6.95 | 42.85, 40.46, 42.76 |
| hires | 27.79 | 63.84, 66.95, 66.89 |
| motion_b2 | 8.95 | 74.35, 72.93, 74.75 |
| random_b1 | 7.95 | 65.44, 64.12, 66.06 |
| random_b2 | 8.95 | 64.63, 62.32, 64.44 |
| random_b3 | 9.95 | 64.26, 61.09, 62.73 |
| random_b4 | 10.95 | 63.43, 59.68, 61.57 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.297 |
| hires | 0.289 |
| random_b2 | 0.308 |
| motion_b2 | 0.381 |
| attention_b2 | 0.340 |
| eyes_b2 | 0.313 |
| eyes_stop | 0.311 |
| best_pair | 0.614 |
| all9 | 0.276 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +6.14 | 152 |
| 2-3 | -5.89 | 713 |
| 4-6 | -3.24 | 1080 |
| 7-10 | -5.05 | 1985 |

## dinov2  (3930 test clips, 3 seeds, rho=0.85, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -2.59 pts [-3.63, -1.33], McNemar p<=4.18e-05 | >= +2 pts | fail |
| 2. Beats random | +0.43 pts [-0.30, +1.20] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -20.07 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.962 (pooled 0.968); adaptive eyes 0.826; greedy finds the best pair in 43% of clips, falls below 3/4 in 2.2% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.13, MC 0.11, closed~MC 0.34 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.006 vs no-R 0.241; glance 0.023 vs 0.202; All-9 0.068 vs 0.698 | <= 0.05 | PASS |

Accuracy: glance 82.08, eyes(2) 79.49, random(2) 79.07, best pair (oracle) 92.34, All-9 62.01

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 85.42 | 27.8 | -5.93 [-7.75, -4.44] |
| B3 attention | 83.41 | 8.9 | -3.92 [-5.07, -3.12] |
| B4 motion | 84.13 | 8.9 | -4.64 [-5.84, -3.66] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 67.10, 63.82, 55.13 |
| all9_noR | 69.48 | 32.29, 26.59, 19.95 |
| attention_b2 | 8.95 | 83.56, 82.98, 83.71 |
| best_pair | 8.95 | 92.24, 92.11, 92.70 |
| eyes_b1 | 7.95 | 81.09, 80.66, 81.75 |
| eyes_b2 | 8.95 | 80.18, 78.35, 79.97 |
| eyes_b3 | 9.95 | 78.37, 75.85, 74.75 |
| eyes_b4 | 10.95 | 75.57, 73.38, 69.66 |
| eyes_mc_b2 | 8.95 | 79.19, 77.66, 79.74 |
| eyes_noR_b2 | 8.95 | 68.96, 67.53, 58.90 |
| eyes_overlapR_b2 | 8.95 | 81.12, 79.01, 79.31 |
| eyes_stop | 8.09 | 79.06, 77.58, 77.09 |
| glance | 6.95 | 82.19, 81.53, 82.54 |
| glance_noR | 6.95 | 70.99, 71.09, 64.70 |
| hires | 27.79 | 84.99, 85.55, 85.75 |
| motion_b2 | 8.95 | 84.02, 83.84, 84.55 |
| random_b1 | 7.95 | 80.58, 80.20, 80.85 |
| random_b2 | 8.95 | 79.70, 78.66, 78.86 |
| random_b3 | 9.95 | 78.63, 77.70, 77.12 |
| random_b4 | 10.95 | 77.73, 76.50, 74.95 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.475 |
| hires | 0.546 |
| random_b2 | 0.470 |
| motion_b2 | 0.522 |
| attention_b2 | 0.527 |
| eyes_b2 | 0.475 |
| eyes_stop | 0.474 |
| best_pair | 0.743 |
| all9 | 0.410 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +10.53 | 152 |
| 2-3 | -4.39 | 713 |
| 4-6 | -2.84 | 1080 |
| 7-10 | -2.80 | 1985 |

## dinov3  (3930 test clips, 3 seeds, rho=0.86, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -3.59 pts [-6.73, -0.73], McNemar p<=3.37e-05 | >= +2 pts | fail |
| 2. Beats random | +0.19 pts [-1.16, +1.45] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -22.01 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.967 (pooled 0.973); adaptive eyes 0.845; greedy finds the best pair in 48% of clips, falls below 3/4 in 2.1% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.11, MC 0.12, closed~MC 0.41 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.016 vs no-R 0.314; glance 0.034 vs 0.257; All-9 0.102 vs 0.703 | <= 0.05 | PASS |

Accuracy: glance 78.89, eyes(2) 75.30, random(2) 75.11, best pair (oracle) 90.91, All-9 56.88

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 82.15 | 27.8 | -6.84 [-9.54, -3.95] |
| B3 attention | 73.27 | 8.9 | +2.04 [+0.63, +3.56] |
| B4 motion | 81.52 | 8.9 | -6.22 [-8.56, -3.88] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 60.15, 57.30, 53.19 |
| all9_noR | 69.48 | 31.81, 24.45, 20.82 |
| attention_b2 | 8.95 | 73.16, 73.36, 73.30 |
| best_pair | 8.95 | 91.02, 90.74, 90.99 |
| eyes_b1 | 7.95 | 76.74, 77.38, 78.60 |
| eyes_b2 | 8.95 | 72.98, 75.60, 77.35 |
| eyes_b3 | 9.95 | 71.02, 73.13, 73.53 |
| eyes_b4 | 10.95 | 67.38, 68.52, 69.94 |
| eyes_mc_b2 | 8.95 | 74.86, 75.14, 76.66 |
| eyes_noR_b2 | 8.95 | 58.60, 57.56, 53.19 |
| eyes_overlapR_b2 | 8.95 | 75.80, 76.62, 76.58 |
| eyes_stop | 8.30 | 73.26, 74.07, 73.86 |
| glance | 6.95 | 78.83, 78.35, 79.51 |
| glance_noR | 6.95 | 61.48, 61.65, 57.90 |
| hires | 27.79 | 82.11, 81.45, 82.90 |
| motion_b2 | 8.95 | 81.50, 81.12, 81.95 |
| random_b1 | 7.95 | 76.99, 76.97, 77.64 |
| random_b2 | 8.95 | 74.88, 74.86, 75.60 |
| random_b3 | 9.95 | 73.65, 73.28, 73.85 |
| random_b4 | 10.95 | 72.16, 71.75, 72.08 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.402 |
| hires | 0.475 |
| random_b2 | 0.412 |
| motion_b2 | 0.479 |
| attention_b2 | 0.388 |
| eyes_b2 | 0.435 |
| eyes_stop | 0.435 |
| best_pair | 0.692 |
| all9 | 0.345 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +4.17 | 152 |
| 2-3 | -3.93 | 713 |
| 4-6 | -2.81 | 1080 |
| 7-10 | -4.48 | 1985 |

## siglip2  (3930 test clips, 3 seeds, rho=0.88, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -5.62 pts [-8.80, -2.49], McNemar p<=1.92e-08 | >= +2 pts | fail |
| 2. Beats random | -1.33 pts [-2.79, +0.03] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -20.08 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.981 (pooled 0.984); adaptive eyes 0.856; greedy finds the best pair in 52% of clips, falls below 3/4 in 0.3% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.15, MC 0.21, closed~MC 0.44 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.019 vs no-R 0.413; glance 0.035 vs 0.358; All-9 0.090 vs 0.774 | <= 0.05 | PASS |

Accuracy: glance 70.41, eyes(2) 64.78, random(2) 66.12, best pair (oracle) 89.19, All-9 50.33

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 68.80 | 27.8 | -4.02 [-11.34, +2.58] |
| B4 motion | 75.95 | 8.9 | -11.17 [-15.74, -7.17] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 56.31, 45.45, 49.25 |
| all9_noR | 69.48 | 20.28, 16.26, 15.65 |
| best_pair | 8.95 | 88.98, 88.93, 89.67 |
| eyes_b1 | 7.95 | 68.45, 67.58, 67.91 |
| eyes_b2 | 8.95 | 66.74, 60.41, 67.22 |
| eyes_b3 | 9.95 | 64.38, 56.03, 64.09 |
| eyes_b4 | 10.95 | 61.40, 52.39, 60.42 |
| eyes_mc_b2 | 8.95 | 64.25, 60.05, 68.01 |
| eyes_noR_b2 | 8.95 | 44.53, 39.59, 41.51 |
| eyes_overlapR_b2 | 8.95 | 67.48, 62.01, 67.65 |
| eyes_stop | 8.88 | 66.03, 61.15, 64.49 |
| glance | 6.95 | 70.41, 70.13, 70.71 |
| glance_noR | 6.95 | 47.81, 45.39, 46.96 |
| hires | 27.79 | 68.98, 67.81, 69.64 |
| motion_b2 | 8.95 | 75.98, 75.39, 76.51 |
| random_b1 | 7.95 | 68.05, 66.77, 68.78 |
| random_b2 | 8.95 | 66.77, 64.46, 67.14 |
| random_b3 | 9.95 | 65.65, 63.14, 65.67 |
| random_b4 | 10.95 | 65.26, 61.41, 64.15 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.311 |
| hires | 0.279 |
| random_b2 | 0.327 |
| motion_b2 | 0.402 |
| eyes_b2 | 0.329 |
| eyes_stop | 0.337 |
| best_pair | 0.678 |
| all9 | 0.291 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +6.80 | 152 |
| 2-3 | -7.06 | 713 |
| 4-6 | -5.06 | 1080 |
| 7-10 | -6.36 | 1985 |
