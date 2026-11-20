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

Accuracy: glance 68.44, eyes(2) 64.16 (eyes), eyes with stopping 61.49 (2.01 looks/clip), random(2) 63.79, best pair (oracle) 87.48, All-9 51.87

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

## clip-boxes-preview-gated-tau  (3930 test clips, 3 seeds, rho=0.81, lam=0.40)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +9.25 pts [+7.32, +11.21], McNemar p<=4.06e-73 | >= +2 pts | PASS |
| 2. Beats random | +6.03 pts [+5.25, +6.93] | >= +1 pt | PASS |
| 3. Gain recovered | 1.16 (All-9 gains +7.96 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.996 (pooled 0.996); adaptive eyes 0.670; greedy finds the best pair in 79% of clips, falls below 3/4 in 0.0% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.31, MC 0.16, closed~MC 0.42 | >= 0.3 | PASS |
| 6. Calibration (ECE) | eyes 0.020 vs no-R 0.347; glance 0.023 vs 0.373; All-9 0.023 vs 0.409 | <= 0.05 | PASS |

Accuracy: glance 67.14, eyes(2) 76.39 (eyesmix), eyes with stopping 76.23 (1.59 looks/clip), random(2) 70.35, best pair (oracle) 83.99, All-9 75.10

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 65.89 | 27.8 | +10.50 [+6.79, +14.60] |
| B3 attention | 70.82 | 8.9 | +5.56 [+4.34, +6.78] |
| B4 motion | 75.75 | 8.9 | +0.64 [+0.01, +1.45] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 76.16, 74.17, 74.98 |
| all9_noR | 69.48 | 60.53, 41.81, 48.10 |
| attention_b2 | 8.95 | 70.28, 71.02, 71.19 |
| best_pair | 8.95 | 85.47, 83.38, 83.15 |
| eyes_b1 | 7.95 | 75.73, 75.45, 75.03 |
| eyes_b2 | 8.95 | 75.88, 75.17, 75.13 |
| eyes_b3 | 9.95 | 75.85, 75.06, 75.34 |
| eyes_b4 | 10.95 | 76.11, 75.19, 75.29 |
| eyes_mc_b2 | 8.95 | 75.06, 75.04, 75.41 |
| eyes_noR_b2 | 8.95 | 52.29, 47.18, 49.17 |
| eyes_overlapR_b2 | 8.95 | 76.49, 75.73, 76.02 |
| eyes_stop | 7.98 | 75.88, 75.60, 74.65 |
| eyesmix_b1 | 7.95 | 75.57, 75.14, 74.88 |
| eyesmix_b2 | 8.95 | 76.74, 76.01, 76.43 |
| eyesmix_b3 | 9.95 | 77.10, 75.95, 76.51 |
| eyesmix_b4 | 10.95 | 77.53, 75.88, 76.48 |
| eyesmix_stop | 8.53 | 77.38, 75.47, 75.85 |
| glance | 6.95 | 66.54, 67.20, 67.70 |
| glance_noR | 6.95 | 42.85, 40.46, 42.76 |
| hires | 27.79 | 63.84, 66.95, 66.89 |
| motion_b2 | 8.95 | 76.64, 75.29, 75.34 |
| random_b1 | 7.95 | 68.47, 68.86, 69.64 |
| random_b2 | 8.95 | 69.85, 70.30, 70.93 |
| random_b3 | 9.95 | 71.24, 71.18, 71.76 |
| random_b4 | 10.95 | 71.90, 71.96, 72.61 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.305 |
| hires | 0.289 |
| random_b2 | 0.342 |
| motion_b2 | 0.398 |
| attention_b2 | 0.346 |
| eyes_b2 | 0.398 |
| eyes_stop | 0.401 |
| eyesmix_b2 | 0.417 |
| eyesmix_stop | 0.416 |
| best_pair | 0.541 |
| all9 | 0.406 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +8.99 | 152 |
| 2-3 | +9.96 | 713 |
| 4-6 | +8.21 | 1080 |
| 7-10 | +9.57 | 1985 |

## dinov2  (3930 test clips, 3 seeds, rho=0.85, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -2.59 pts [-3.63, -1.33], McNemar p<=4.18e-05 | >= +2 pts | fail |
| 2. Beats random | +0.43 pts [-0.30, +1.20] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -20.07 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.962 (pooled 0.968); adaptive eyes 0.826; greedy finds the best pair in 43% of clips, falls below 3/4 in 2.2% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.13, MC 0.11, closed~MC 0.34 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.006 vs no-R 0.241; glance 0.023 vs 0.202; All-9 0.068 vs 0.698 | <= 0.05 | PASS |

Accuracy: glance 82.08, eyes(2) 79.49 (eyes), eyes with stopping 77.91 (1.14 looks/clip), random(2) 79.07, best pair (oracle) 92.34, All-9 62.01

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

## dinov2-boxes-preview-gated-tau  (3930 test clips, 3 seeds, rho=0.76, lam=0.53)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +4.49 pts [+3.14, +6.33], McNemar p<=8.67e-33 | >= +2 pts | PASS |
| 2. Beats random | +3.16 pts [+2.18, +4.39] | >= +1 pt | PASS |
| 3. Gain recovered | 1.01 (All-9 gains +4.43 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.996 (pooled 0.997); adaptive eyes 0.689; greedy finds the best pair in 75% of clips, falls below 3/4 in 0.0% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.28, MC 0.17, closed~MC 0.36 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.041 vs no-R 0.167; glance 0.016 vs 0.202; All-9 0.044 vs 0.156 | <= 0.05 | PASS |

Accuracy: glance 81.60, eyes(2) 86.09 (eyesmix), eyes with stopping 86.32 (0.91 looks/clip), random(2) 82.93, best pair (oracle) 89.79, All-9 86.03

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 85.42 | 27.8 | +0.67 [-0.04, +1.63] |
| B3 attention | 85.07 | 8.9 | +1.02 [+0.50, +1.63] |
| B4 motion | 85.49 | 8.9 | +0.60 [-0.04, +1.50] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 86.41, 85.70, 86.00 |
| all9_noR | 69.48 | 84.05, 78.42, 79.79 |
| attention_b2 | 8.95 | 85.62, 84.53, 85.09 |
| best_pair | 8.95 | 90.64, 89.36, 89.39 |
| eyes_b1 | 7.95 | 85.90, 85.73, 85.47 |
| eyes_b2 | 8.95 | 86.26, 85.62, 85.59 |
| eyes_b3 | 9.95 | 86.54, 85.75, 85.65 |
| eyes_b4 | 10.95 | 86.54, 85.78, 85.62 |
| eyes_mc_b2 | 8.95 | 86.06, 85.34, 85.54 |
| eyes_noR_b2 | 8.95 | 77.68, 76.87, 73.20 |
| eyes_overlapR_b2 | 8.95 | 86.49, 86.03, 86.26 |
| eyes_stop | 7.82 | 86.44, 85.70, 85.54 |
| eyesmix_b1 | 7.95 | 86.16, 85.09, 85.72 |
| eyesmix_b2 | 8.95 | 86.67, 85.65, 85.98 |
| eyesmix_b3 | 9.95 | 86.92, 85.70, 86.21 |
| eyesmix_b4 | 10.95 | 86.97, 85.98, 86.33 |
| eyesmix_stop | 7.85 | 87.02, 85.85, 86.10 |
| glance | 6.95 | 81.76, 81.07, 82.01 |
| glance_noR | 6.95 | 70.99, 71.09, 64.70 |
| hires | 27.79 | 84.99, 85.55, 85.75 |
| motion_b2 | 8.95 | 85.93, 84.96, 85.59 |
| random_b1 | 7.95 | 82.69, 81.80, 82.72 |
| random_b2 | 8.95 | 83.15, 82.48, 83.19 |
| random_b3 | 9.95 | 83.65, 83.03, 83.63 |
| random_b4 | 10.95 | 83.96, 83.49, 83.91 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.483 |
| hires | 0.546 |
| random_b2 | 0.522 |
| motion_b2 | 0.560 |
| attention_b2 | 0.568 |
| eyes_b2 | 0.569 |
| eyes_stop | 0.569 |
| eyesmix_b2 | 0.586 |
| eyesmix_stop | 0.587 |
| best_pair | 0.667 |
| all9 | 0.584 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +10.75 | 152 |
| 2-3 | +5.75 | 713 |
| 4-6 | +3.98 | 1080 |
| 7-10 | +3.83 | 1985 |

## dinov3  (3930 test clips, 3 seeds, rho=0.86, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -3.59 pts [-6.73, -0.73], McNemar p<=3.37e-05 | >= +2 pts | fail |
| 2. Beats random | +0.19 pts [-1.16, +1.45] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -22.01 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.967 (pooled 0.973); adaptive eyes 0.845; greedy finds the best pair in 48% of clips, falls below 3/4 in 2.1% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.11, MC 0.12, closed~MC 0.41 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.016 vs no-R 0.314; glance 0.034 vs 0.257; All-9 0.102 vs 0.703 | <= 0.05 | PASS |

Accuracy: glance 78.89, eyes(2) 75.30 (eyes), eyes with stopping 73.72 (1.35 looks/clip), random(2) 75.11, best pair (oracle) 90.91, All-9 56.88

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 82.15 | 27.8 | -6.84 [-9.54, -3.95] |
| B3 attention | 83.60 | 8.9 | -8.30 [-10.54, -5.87] |
| B4 motion | 81.52 | 8.9 | -6.22 [-8.56, -3.88] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 60.15, 57.30, 53.19 |
| all9_noR | 69.48 | 31.81, 24.45, 20.82 |
| attention_b2 | 8.95 | 83.69, 83.16, 83.97 |
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
| attention_b2 | 0.531 |
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

## dinov3-boxes-preview-gated-tau  (3930 test clips, 3 seeds, rho=0.78, lam=0.60)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +5.84 pts [+4.18, +8.22], McNemar p<=3.63e-42 | >= +2 pts | PASS |
| 2. Beats random | +3.91 pts [+2.82, +5.38] | >= +1 pt | PASS |
| 3. Gain recovered | 0.97 (All-9 gains +6.00 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.997 (pooled 0.998); adaptive eyes 0.687; greedy finds the best pair in 78% of clips, falls below 3/4 in 0.0% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.34, MC 0.22, closed~MC 0.36 | >= 0.3 | PASS |
| 6. Calibration (ECE) | eyes 0.040 vs no-R 0.229; glance 0.019 vs 0.257; All-9 0.037 vs 0.220 | <= 0.05 | PASS |

Accuracy: glance 78.23, eyes(2) 84.06 (eyesmix), eyes with stopping 84.13 (1.05 looks/clip), random(2) 80.15, best pair (oracle) 88.60, All-9 84.22

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 82.15 | 27.8 | +1.92 [+0.80, +3.62] |
| B3 attention | 84.73 | 8.9 | -0.67 [-1.26, -0.02] |
| B4 motion | 83.01 | 8.9 | +1.05 [+0.27, +2.50] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 85.11, 83.97, 83.61 |
| all9_noR | 69.48 | 79.16, 69.34, 72.89 |
| attention_b2 | 8.95 | 85.11, 84.43, 84.68 |
| best_pair | 8.95 | 89.11, 88.37, 88.34 |
| eyes_b1 | 7.95 | 84.27, 84.02, 83.56 |
| eyes_b2 | 8.95 | 84.81, 84.02, 83.63 |
| eyes_b3 | 9.95 | 84.76, 83.99, 83.69 |
| eyes_b4 | 10.95 | 84.89, 84.05, 83.66 |
| eyes_mc_b2 | 8.95 | 84.58, 84.20, 83.69 |
| eyes_noR_b2 | 8.95 | 70.18, 67.81, 65.77 |
| eyes_overlapR_b2 | 8.95 | 84.94, 83.72, 84.53 |
| eyes_stop | 8.01 | 84.94, 84.17, 83.61 |
| eyesmix_b1 | 7.95 | 84.22, 83.54, 83.61 |
| eyesmix_b2 | 8.95 | 84.61, 83.82, 83.79 |
| eyesmix_b3 | 9.95 | 84.89, 83.87, 83.94 |
| eyesmix_b4 | 10.95 | 85.04, 83.87, 84.12 |
| eyesmix_stop | 7.99 | 84.91, 83.72, 83.79 |
| glance | 6.95 | 78.47, 77.74, 78.49 |
| glance_noR | 6.95 | 61.48, 61.65, 57.90 |
| hires | 27.79 | 82.11, 81.45, 82.90 |
| motion_b2 | 8.95 | 83.46, 82.42, 83.18 |
| random_b1 | 7.95 | 79.47, 78.92, 79.55 |
| random_b2 | 8.95 | 80.31, 79.81, 80.35 |
| random_b3 | 9.95 | 81.07, 80.41, 80.94 |
| random_b4 | 10.95 | 81.61, 80.81, 81.31 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.403 |
| hires | 0.475 |
| random_b2 | 0.464 |
| motion_b2 | 0.505 |
| attention_b2 | 0.551 |
| eyes_b2 | 0.531 |
| eyes_stop | 0.535 |
| eyesmix_b2 | 0.544 |
| eyesmix_stop | 0.548 |
| best_pair | 0.626 |
| all9 | 0.545 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +14.47 | 152 |
| 2-3 | +8.70 | 713 |
| 4-6 | +3.89 | 1080 |
| 7-10 | +5.21 | 1985 |

## siglip2  (3930 test clips, 3 seeds, rho=0.88, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | -5.62 pts [-8.80, -2.49], McNemar p<=1.92e-08 | >= +2 pts | fail |
| 2. Beats random | -1.33 pts [-2.79, +0.03] | >= +1 pt | fail |
| 3. Gain recovered | nan (All-9 gains -20.08 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.981 (pooled 0.984); adaptive eyes 0.856; greedy finds the best pair in 52% of clips, falls below 3/4 in 0.3% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.15, MC 0.21, closed~MC 0.44 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.019 vs no-R 0.413; glance 0.035 vs 0.358; All-9 0.090 vs 0.774 | <= 0.05 | PASS |

Accuracy: glance 70.41, eyes(2) 64.78 (eyes), eyes with stopping 63.88 (1.94 looks/clip), random(2) 66.12, best pair (oracle) 89.19, All-9 50.33

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

## siglip2-boxes-preview-gated-tau  (3930 test clips, 3 seeds, rho=0.81, lam=0.40)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +9.63 pts [+7.53, +11.84], McNemar p<=9.77e-73 | >= +2 pts | PASS |
| 2. Beats random | +6.10 pts [+5.30, +7.09] | >= +1 pt | PASS |
| 3. Gain recovered | 1.16 (All-9 gains +8.33 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.997 (pooled 0.998); adaptive eyes 0.663; greedy finds the best pair in 81% of clips, falls below 3/4 in 0.0% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.35, MC 0.16, closed~MC 0.40 | >= 0.3 | PASS |
| 6. Calibration (ECE) | eyes 0.031 vs no-R 0.317; glance 0.012 vs 0.358; All-9 0.032 vs 0.298 | <= 0.05 | PASS |

Accuracy: glance 69.28, eyes(2) 78.91 (eyesmix), eyes with stopping 78.87 (1.15 looks/clip), random(2) 72.81, best pair (oracle) 85.66, All-9 77.61

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 68.80 | 27.8 | +10.10 [+6.92, +14.31] |
| B4 motion | 78.19 | 8.9 | +0.71 [+0.08, +1.76] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 78.73, 76.03, 78.09 |
| all9_noR | 69.48 | 71.37, 53.41, 64.04 |
| best_pair | 8.95 | 86.74, 84.43, 85.82 |
| eyes_b1 | 7.95 | 78.83, 77.43, 77.96 |
| eyes_b2 | 8.95 | 79.59, 77.40, 78.42 |
| eyes_b3 | 9.95 | 79.64, 77.56, 78.54 |
| eyes_b4 | 10.95 | 79.64, 77.63, 78.65 |
| eyes_mc_b2 | 8.95 | 79.08, 77.56, 78.04 |
| eyes_noR_b2 | 8.95 | 58.12, 52.04, 57.19 |
| eyes_overlapR_b2 | 8.95 | 79.77, 78.42, 79.92 |
| eyes_stop | 8.01 | 79.13, 77.58, 78.44 |
| eyesmix_b1 | 7.95 | 78.47, 77.15, 78.39 |
| eyesmix_b2 | 8.95 | 79.57, 77.86, 79.31 |
| eyesmix_b3 | 9.95 | 79.85, 78.04, 79.38 |
| eyesmix_b4 | 10.95 | 80.03, 78.24, 79.26 |
| eyesmix_stop | 8.10 | 79.57, 77.89, 79.18 |
| glance | 6.95 | 68.68, 69.29, 69.89 |
| glance_noR | 6.95 | 47.81, 45.39, 46.96 |
| hires | 27.79 | 68.98, 67.81, 69.64 |
| motion_b2 | 8.95 | 79.03, 77.43, 78.14 |
| random_b1 | 7.95 | 71.08, 71.11, 71.98 |
| random_b2 | 8.95 | 72.91, 72.19, 73.34 |
| random_b3 | 9.95 | 74.03, 73.24, 74.04 |
| random_b4 | 10.95 | 74.73, 73.91, 74.77 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.313 |
| hires | 0.279 |
| random_b2 | 0.365 |
| motion_b2 | 0.436 |
| eyes_b2 | 0.449 |
| eyes_stop | 0.453 |
| eyesmix_b2 | 0.456 |
| eyesmix_stop | 0.457 |
| best_pair | 0.586 |
| all9 | 0.441 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +12.28 | 152 |
| 2-3 | +10.38 | 713 |
| 4-6 | +7.25 | 1080 |
| 7-10 | +10.45 | 1985 |
