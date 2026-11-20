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

## clip-boxes  (3930 test clips, 3 seeds, rho=0.81, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +5.80 pts [+3.35, +8.09], McNemar p<=1.41e-13 | >= +2 pts | PASS |
| 2. Beats random | +5.37 pts [+3.61, +6.98] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -22.13 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.981 (pooled 0.981); adaptive eyes 0.906; greedy finds the best pair in 41% of clips, falls below 3/4 in 1.6% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.13, MC 0.09, closed~MC 0.25 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.053 vs no-R 0.355; glance 0.021 vs 0.373; All-9 0.298 vs 0.614 | <= 0.05 | fail |

Accuracy: glance 67.05, eyes(2) 72.85, random(2) 67.48, best pair (oracle) 87.68, All-9 44.92

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 65.89 | 27.8 | +6.96 [+2.69, +11.31] |
| B3 attention | 68.24 | 8.9 | +4.61 [+1.67, +7.95] |
| B4 motion | 75.83 | 8.9 | -2.98 [-3.90, -1.89] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 61.17, 33.05, 40.54 |
| all9_noR | 69.48 | 40.25, 19.95, 24.59 |
| attention_b2 | 8.95 | 68.75, 67.61, 68.39 |
| best_pair | 8.95 | 87.68, 87.23, 88.14 |
| eyes_b1 | 7.95 | 73.13, 73.59, 75.18 |
| eyes_b2 | 8.95 | 73.56, 71.42, 73.58 |
| eyes_b3 | 9.95 | 73.61, 67.38, 70.25 |
| eyes_b4 | 10.95 | 72.26, 60.56, 65.21 |
| eyes_mc_b2 | 8.95 | 75.06, 71.45, 73.15 |
| eyes_noR_b2 | 8.95 | 50.99, 45.95, 48.56 |
| eyes_overlapR_b2 | 8.95 | 75.85, 73.77, 75.06 |
| eyes_stop | 8.92 | 73.13, 65.22, 67.91 |
| glance | 6.95 | 66.54, 66.79, 67.83 |
| glance_noR | 6.95 | 42.85, 40.46, 42.76 |
| hires | 27.79 | 63.84, 66.95, 66.89 |
| motion_b2 | 8.95 | 76.26, 75.32, 75.92 |
| random_b1 | 7.95 | 67.53, 66.89, 68.44 |
| random_b2 | 8.95 | 68.53, 65.95, 67.96 |
| random_b3 | 9.95 | 69.51, 64.41, 66.24 |
| random_b4 | 10.95 | 69.44, 61.88, 64.19 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.306 |
| hires | 0.289 |
| random_b2 | 0.337 |
| motion_b2 | 0.401 |
| attention_b2 | 0.348 |
| eyes_b2 | 0.398 |
| eyes_stop | 0.399 |
| best_pair | 0.627 |
| all9 | 0.295 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +3.29 | 152 |
| 2-3 | +3.32 | 713 |
| 4-6 | +4.44 | 1080 |
| 7-10 | +7.62 | 1985 |

## clip-boxes-preview  (3930 test clips, 3 seeds, rho=0.81, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +6.62 pts [+4.03, +9.19], McNemar p<=1.23e-21 | >= +2 pts | PASS |
| 2. Beats random | +6.20 pts [+4.30, +8.11] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -22.13 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.979 (pooled 0.978); adaptive eyes 0.911; greedy finds the best pair in 44% of clips, falls below 3/4 in 1.8% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.17, MC 0.11, closed~MC 0.29 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.053 vs no-R 0.357; glance 0.021 vs 0.373; All-9 0.298 vs 0.614 | <= 0.05 | fail |

Accuracy: glance 67.05, eyes(2) 73.67, random(2) 67.48, best pair (oracle) 87.68, All-9 44.92

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 65.89 | 27.8 | +7.79 [+3.39, +12.50] |
| B3 attention | 68.24 | 8.9 | +5.43 [+2.38, +9.10] |
| B4 motion | 75.83 | 8.9 | -2.15 [-3.17, -1.06] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 61.17, 33.05, 40.54 |
| all9_noR | 69.48 | 40.25, 19.95, 24.59 |
| attention_b2 | 8.95 | 68.75, 67.61, 68.39 |
| best_pair | 8.95 | 87.68, 87.23, 88.14 |
| eyes_b1 | 7.95 | 75.06, 75.67, 75.62 |
| eyes_b2 | 8.95 | 74.66, 72.88, 73.50 |
| eyes_b3 | 9.95 | 73.92, 67.53, 68.85 |
| eyes_b4 | 10.95 | 71.91, 60.23, 63.58 |
| eyes_mc_b2 | 8.95 | 73.41, 69.77, 71.49 |
| eyes_noR_b2 | 8.95 | 51.76, 45.85, 48.08 |
| eyes_overlapR_b2 | 8.95 | 76.26, 73.89, 74.60 |
| eyes_stop | 8.81 | 73.31, 65.95, 67.45 |
| eyesmix_b1 | 7.95 | 72.88, 71.32, 73.35 |
| eyesmix_b2 | 8.95 | 73.79, 69.31, 71.70 |
| eyesmix_b3 | 9.95 | 72.82, 64.22, 68.29 |
| eyesmix_b4 | 10.95 | 71.35, 59.34, 63.71 |
| eyesmix_stop | 9.11 | 72.11, 61.55, 65.69 |
| glance | 6.95 | 66.54, 66.79, 67.83 |
| glance_noR | 6.95 | 42.85, 40.46, 42.76 |
| hires | 27.79 | 63.84, 66.95, 66.89 |
| motion_b2 | 8.95 | 76.26, 75.32, 75.92 |
| random_b1 | 7.95 | 67.53, 66.89, 68.44 |
| random_b2 | 8.95 | 68.53, 65.95, 67.96 |
| random_b3 | 9.95 | 69.51, 64.41, 66.24 |
| random_b4 | 10.95 | 69.44, 61.88, 64.19 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.306 |
| hires | 0.289 |
| random_b2 | 0.337 |
| motion_b2 | 0.401 |
| attention_b2 | 0.348 |
| eyes_b2 | 0.413 |
| eyes_stop | 0.409 |
| best_pair | 0.627 |
| all9 | 0.295 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +5.48 | 152 |
| 2-3 | +4.63 | 713 |
| 4-6 | +5.12 | 1080 |
| 7-10 | +8.25 | 1985 |

## clip-boxes-preview-shaped  (3930 test clips, 3 seeds, rho=0.83, lam=0.43)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +6.40 pts [+4.60, +8.14], McNemar p<=3.77e-32 | >= +2 pts | PASS |
| 2. Beats random | +3.27 pts [+2.58, +3.75] | >= +1 pt | PASS |
| 3. Gain recovered | 0.75 (All-9 gains +8.53 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.967 (pooled 0.968); adaptive eyes 0.845; greedy finds the best pair in 64% of clips, falls below 3/4 in 1.4% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.32, MC 0.29, closed~MC 0.71 | >= 0.3 | PASS |
| 6. Calibration (ECE) | eyes 0.025 vs no-R 0.350; glance 0.029 vs 0.373; All-9 0.037 vs 0.323 | <= 0.05 | PASS |

Accuracy: glance 67.64, eyes(2) 74.05, random(2) 70.78, best pair (oracle) 82.47, All-9 76.17

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 65.89 | 27.8 | +8.16 [+4.07, +12.60] |
| B3 attention | 71.53 | 8.9 | +2.52 [+1.47, +3.41] |
| B4 motion | 74.30 | 8.9 | -0.25 [-0.93, +0.37] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 75.85, 76.01, 76.69 |
| all9_noR | 69.48 | 45.04, 66.59, 69.87 |
| attention_b2 | 8.95 | 71.15, 71.55, 71.90 |
| best_pair | 8.95 | 82.26, 82.70, 82.46 |
| eyes_b1 | 7.95 | 72.95, 71.48, 71.90 |
| eyes_b2 | 8.95 | 75.24, 73.10, 73.81 |
| eyes_b3 | 9.95 | 75.34, 73.87, 74.57 |
| eyes_b4 | 10.95 | 75.67, 74.02, 74.68 |
| eyes_mc_b2 | 8.95 | 75.24, 73.08, 73.50 |
| eyes_noR_b2 | 8.95 | 47.66, 47.25, 49.12 |
| eyes_overlapR_b2 | 8.95 | 76.11, 74.61, 74.96 |
| eyes_stop | 9.31 | 75.67, 74.05, 74.68 |
| eyesmix_b1 | 7.95 | 72.47, 71.86, 72.97 |
| eyesmix_b2 | 8.95 | 73.94, 73.33, 74.29 |
| eyesmix_b3 | 9.95 | 74.48, 74.17, 75.54 |
| eyesmix_b4 | 10.95 | 74.99, 74.94, 76.30 |
| eyesmix_stop | 8.71 | 74.73, 74.07, 75.74 |
| glance | 6.95 | 67.23, 67.51, 68.21 |
| glance_noR | 6.95 | 42.85, 40.46, 42.76 |
| hires | 27.79 | 63.84, 66.95, 66.89 |
| motion_b2 | 8.95 | 74.58, 74.20, 74.14 |
| random_b1 | 7.95 | 69.15, 69.19, 70.16 |
| random_b2 | 8.95 | 70.33, 70.47, 71.55 |
| random_b3 | 9.95 | 71.41, 71.49, 72.48 |
| random_b4 | 10.95 | 72.14, 72.18, 72.99 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.302 |
| hires | 0.289 |
| random_b2 | 0.317 |
| motion_b2 | 0.346 |
| attention_b2 | 0.326 |
| eyes_b2 | 0.344 |
| eyes_stop | 0.353 |
| best_pair | 0.480 |
| all9 | 0.346 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | -0.00 | 152 |
| 2-3 | +5.52 | 713 |
| 4-6 | +6.42 | 1080 |
| 7-10 | +7.20 | 1985 |

## clip-boxes-where  (3930 test clips, 3 seeds, rho=0.81, lam=0.00)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +6.80 pts [+4.13, +9.55], McNemar p<=3.51e-23 | >= +2 pts | PASS |
| 2. Beats random | +6.37 pts [+4.39, +8.48] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -22.13 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.979 (pooled 0.977); adaptive eyes 0.911; greedy finds the best pair in 43% of clips, falls below 3/4 in 1.8% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.17, MC 0.11, closed~MC 0.29 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.053 vs no-R 0.356; glance 0.021 vs 0.373; All-9 0.298 vs 0.614 | <= 0.05 | fail |

Accuracy: glance 67.05, eyes(2) 73.85, random(2) 67.48, best pair (oracle) 87.68, All-9 44.92

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 65.89 | 27.8 | +7.96 [+3.50, +12.84] |
| B3 attention | 68.24 | 8.9 | +5.61 [+2.42, +9.48] |
| B4 motion | 75.83 | 8.9 | -1.98 [-3.01, -0.89] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 61.17, 33.05, 40.54 |
| all9_noR | 69.48 | 40.25, 19.95, 24.59 |
| attention_b2 | 8.95 | 68.75, 67.61, 68.39 |
| best_pair | 8.95 | 87.68, 87.23, 88.14 |
| eyes_b1 | 7.95 | 74.76, 75.73, 75.64 |
| eyes_b2 | 8.95 | 74.73, 73.18, 73.66 |
| eyes_b3 | 9.95 | 74.07, 67.61, 69.05 |
| eyes_b4 | 10.95 | 72.26, 60.51, 63.93 |
| eyes_mc_b2 | 8.95 | 73.26, 70.20, 71.80 |
| eyes_noR_b2 | 8.95 | 51.83, 45.80, 48.15 |
| eyes_overlapR_b2 | 8.95 | 76.36, 73.56, 74.90 |
| eyes_stop | 8.80 | 73.44, 66.36, 67.75 |
| glance | 6.95 | 66.54, 66.79, 67.83 |
| glance_noR | 6.95 | 42.85, 40.46, 42.76 |
| hires | 27.79 | 63.84, 66.95, 66.89 |
| motion_b2 | 8.95 | 76.26, 75.32, 75.92 |
| random_b1 | 7.95 | 67.53, 66.89, 68.44 |
| random_b2 | 8.95 | 68.53, 65.95, 67.96 |
| random_b3 | 9.95 | 69.51, 64.41, 66.24 |
| random_b4 | 10.95 | 69.44, 61.88, 64.19 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.306 |
| hires | 0.289 |
| random_b2 | 0.337 |
| motion_b2 | 0.401 |
| attention_b2 | 0.348 |
| eyes_b2 | 0.416 |
| eyes_stop | 0.406 |
| best_pair | 0.627 |
| all9 | 0.295 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +5.70 | 152 |
| 2-3 | +5.24 | 713 |
| 4-6 | +4.94 | 1080 |
| 7-10 | +8.46 | 1985 |

## clip-boxes-where-shaped  (3930 test clips, 3 seeds, rho=0.83, lam=0.43)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +6.24 pts [+4.46, +7.99], McNemar p<=8.24e-30 | >= +2 pts | PASS |
| 2. Beats random | +3.11 pts [+2.44, +3.58] | >= +1 pt | PASS |
| 3. Gain recovered | 0.73 (All-9 gains +8.53 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.967 (pooled 0.969); adaptive eyes 0.844; greedy finds the best pair in 64% of clips, falls below 3/4 in 1.4% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.32, MC 0.29, closed~MC 0.71 | >= 0.3 | PASS |
| 6. Calibration (ECE) | eyes 0.025 vs no-R 0.349; glance 0.029 vs 0.373; All-9 0.037 vs 0.323 | <= 0.05 | PASS |

Accuracy: glance 67.64, eyes(2) 73.88, random(2) 70.78, best pair (oracle) 82.47, All-9 76.17

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 65.89 | 27.8 | +8.00 [+3.94, +12.43] |
| B3 attention | 71.53 | 8.9 | +2.36 [+1.34, +3.24] |
| B4 motion | 74.30 | 8.9 | -0.42 [-1.03, +0.17] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 75.85, 76.01, 76.69 |
| all9_noR | 69.48 | 45.04, 66.59, 69.87 |
| attention_b2 | 8.95 | 71.15, 71.55, 71.90 |
| best_pair | 8.95 | 82.26, 82.70, 82.46 |
| eyes_b1 | 7.95 | 73.08, 71.76, 71.57 |
| eyes_b2 | 8.95 | 75.22, 72.90, 73.56 |
| eyes_b3 | 9.95 | 75.24, 74.12, 74.55 |
| eyes_b4 | 10.95 | 75.75, 74.02, 74.73 |
| eyes_mc_b2 | 8.95 | 75.09, 73.16, 73.40 |
| eyes_noR_b2 | 8.95 | 47.86, 47.15, 49.22 |
| eyes_overlapR_b2 | 8.95 | 76.23, 74.94, 75.11 |
| eyes_stop | 9.31 | 75.75, 74.10, 74.73 |
| glance | 6.95 | 67.23, 67.51, 68.21 |
| glance_noR | 6.95 | 42.85, 40.46, 42.76 |
| hires | 27.79 | 63.84, 66.95, 66.89 |
| motion_b2 | 8.95 | 74.58, 74.20, 74.14 |
| random_b1 | 7.95 | 69.15, 69.19, 70.16 |
| random_b2 | 8.95 | 70.33, 70.47, 71.55 |
| random_b3 | 9.95 | 71.41, 71.49, 72.48 |
| random_b4 | 10.95 | 72.14, 72.18, 72.99 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.302 |
| hires | 0.289 |
| random_b2 | 0.317 |
| motion_b2 | 0.346 |
| attention_b2 | 0.326 |
| eyes_b2 | 0.342 |
| eyes_stop | 0.350 |
| best_pair | 0.480 |
| all9 | 0.346 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | -0.66 | 152 |
| 2-3 | +5.10 | 713 |
| 4-6 | +6.45 | 1080 |
| 7-10 | +7.07 | 1985 |

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

## dinov2-boxes  (3930 test clips, 3 seeds, rho=0.79, lam=0.10)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +3.84 pts [+2.47, +5.97], McNemar p<=3.29e-15 | >= +2 pts | PASS |
| 2. Beats random | +3.21 pts [+2.26, +4.65] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -4.68 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.983 (pooled 0.974); adaptive eyes 0.883; greedy finds the best pair in 42% of clips, falls below 3/4 in 1.6% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.13, MC 0.01, closed~MC 0.07 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.039 vs no-R 0.173; glance 0.016 vs 0.202; All-9 0.106 vs 0.293 | <= 0.05 | PASS |

Accuracy: glance 81.82, eyes(2) 85.66, random(2) 82.44, best pair (oracle) 92.00, All-9 77.13

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 85.42 | 27.8 | +0.24 [-0.54, +1.38] |
| B3 attention | 85.19 | 8.9 | +0.47 [-0.07, +1.13] |
| B4 motion | 85.53 | 8.9 | +0.13 [-0.55, +1.18] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 83.84, 72.04, 75.54 |
| all9_noR | 69.48 | 76.87, 60.69, 61.49 |
| attention_b2 | 8.95 | 85.52, 84.48, 85.59 |
| best_pair | 8.95 | 92.04, 91.73, 92.24 |
| eyes_b1 | 7.95 | 85.42, 85.32, 85.44 |
| eyes_b2 | 8.95 | 86.36, 84.94, 85.70 |
| eyes_b3 | 9.95 | 85.93, 84.17, 85.39 |
| eyes_b4 | 10.95 | 86.16, 82.44, 83.91 |
| eyes_mc_b2 | 8.95 | 86.08, 84.89, 85.44 |
| eyes_noR_b2 | 8.95 | 77.10, 76.16, 72.79 |
| eyes_overlapR_b2 | 8.95 | 86.36, 86.49, 86.49 |
| eyes_stop | 7.98 | 86.39, 83.92, 84.98 |
| glance | 6.95 | 82.01, 81.15, 82.31 |
| glance_noR | 6.95 | 70.99, 71.09, 64.70 |
| hires | 27.79 | 84.99, 85.55, 85.75 |
| motion_b2 | 8.95 | 85.73, 85.09, 85.80 |
| random_b1 | 7.95 | 82.58, 81.52, 82.68 |
| random_b2 | 8.95 | 82.87, 81.76, 82.72 |
| random_b3 | 9.95 | 83.32, 81.58, 82.65 |
| random_b4 | 10.95 | 83.53, 81.36, 82.43 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.481 |
| hires | 0.546 |
| random_b2 | 0.517 |
| motion_b2 | 0.571 |
| attention_b2 | 0.577 |
| eyes_b2 | 0.578 |
| eyes_stop | 0.594 |
| best_pair | 0.735 |
| all9 | 0.524 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +9.43 | 152 |
| 2-3 | +4.68 | 713 |
| 4-6 | +3.06 | 1080 |
| 7-10 | +3.54 | 1985 |

## dinov2-boxes-preview  (3930 test clips, 3 seeds, rho=0.79, lam=0.10)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +3.77 pts [+2.24, +6.10], McNemar p<=6.05e-15 | >= +2 pts | PASS |
| 2. Beats random | +3.14 pts [+2.06, +4.75] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -4.68 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.981 (pooled 0.973); adaptive eyes 0.894; greedy finds the best pair in 43% of clips, falls below 3/4 in 1.9% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.16, MC 0.01, closed~MC 0.11 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.044 vs no-R 0.174; glance 0.016 vs 0.202; All-9 0.106 vs 0.293 | <= 0.05 | PASS |

Accuracy: glance 81.82, eyes(2) 85.58, random(2) 82.44, best pair (oracle) 92.00, All-9 77.13

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 85.42 | 27.8 | +0.16 [-0.77, +1.52] |
| B3 attention | 85.19 | 8.9 | +0.39 [-0.30, +1.27] |
| B4 motion | 85.53 | 8.9 | +0.05 [-0.69, +1.18] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 83.84, 72.04, 75.54 |
| all9_noR | 69.48 | 76.87, 60.69, 61.49 |
| attention_b2 | 8.95 | 85.52, 84.48, 85.59 |
| best_pair | 8.95 | 92.04, 91.73, 92.24 |
| eyes_b1 | 7.95 | 85.70, 85.57, 85.70 |
| eyes_b2 | 8.95 | 86.16, 84.94, 85.67 |
| eyes_b3 | 9.95 | 86.31, 83.56, 85.44 |
| eyes_b4 | 10.95 | 86.21, 82.60, 83.76 |
| eyes_mc_b2 | 8.95 | 85.32, 83.94, 85.34 |
| eyes_noR_b2 | 8.95 | 77.40, 76.13, 72.36 |
| eyes_overlapR_b2 | 8.95 | 86.54, 86.23, 86.71 |
| eyes_stop | 7.91 | 86.31, 84.07, 84.88 |
| eyesmix_b1 | 7.95 | 86.31, 84.94, 86.03 |
| eyesmix_b2 | 8.95 | 86.51, 84.66, 86.03 |
| eyesmix_b3 | 9.95 | 86.46, 83.94, 85.67 |
| eyesmix_b4 | 10.95 | 86.51, 82.42, 85.16 |
| eyesmix_stop | 7.96 | 86.69, 84.07, 85.70 |
| glance | 6.95 | 82.01, 81.15, 82.31 |
| glance_noR | 6.95 | 70.99, 71.09, 64.70 |
| hires | 27.79 | 84.99, 85.55, 85.75 |
| motion_b2 | 8.95 | 85.73, 85.09, 85.80 |
| random_b1 | 7.95 | 82.58, 81.52, 82.68 |
| random_b2 | 8.95 | 82.87, 81.76, 82.72 |
| random_b3 | 9.95 | 83.32, 81.58, 82.65 |
| random_b4 | 10.95 | 83.53, 81.36, 82.43 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.481 |
| hires | 0.546 |
| random_b2 | 0.517 |
| motion_b2 | 0.571 |
| attention_b2 | 0.577 |
| eyes_b2 | 0.583 |
| eyes_stop | 0.595 |
| best_pair | 0.735 |
| all9 | 0.524 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +9.21 | 152 |
| 2-3 | +5.47 | 713 |
| 4-6 | +2.56 | 1080 |
| 7-10 | +3.39 | 1985 |

## dinov2-boxes-preview-shaped  (3930 test clips, 3 seeds, rho=0.81, lam=0.53)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +2.74 pts [+1.71, +4.04], McNemar p<=2.28e-07 | >= +2 pts | PASS |
| 2. Beats random | +1.59 pts [+0.95, +2.38] | >= +1 pt | PASS |
| 3. Gain recovered | 0.85 (All-9 gains +3.24 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.973 (pooled 0.970); adaptive eyes 0.843; greedy finds the best pair in 66% of clips, falls below 3/4 in 1.2% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.29, MC 0.27, closed~MC 0.71 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.013 vs no-R 0.183; glance 0.017 vs 0.202; All-9 0.008 vs 0.195 | <= 0.05 | PASS |

Accuracy: glance 81.93, eyes(2) 84.67, random(2) 83.08, best pair (oracle) 89.09, All-9 85.17

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 85.42 | 27.8 | -0.75 [-1.42, -0.26] |
| B3 attention | 84.58 | 8.9 | +0.09 [-0.32, +0.51] |
| B4 motion | 84.62 | 8.9 | +0.05 [-0.41, +0.70] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 85.62, 85.14, 84.78 |
| all9_noR | 69.48 | 78.58, 81.17, 66.79 |
| attention_b2 | 8.95 | 84.58, 84.05, 85.14 |
| best_pair | 8.95 | 89.21, 88.58, 89.51 |
| eyes_b1 | 7.95 | 84.81, 83.59, 84.02 |
| eyes_b2 | 8.95 | 85.32, 84.55, 84.17 |
| eyes_b3 | 9.95 | 85.55, 84.53, 84.68 |
| eyes_b4 | 10.95 | 85.80, 84.83, 84.91 |
| eyes_mc_b2 | 8.95 | 85.34, 84.25, 84.37 |
| eyes_noR_b2 | 8.95 | 75.29, 75.42, 68.82 |
| eyes_overlapR_b2 | 8.95 | 86.16, 85.27, 86.08 |
| eyes_stop | 8.07 | 85.52, 84.81, 84.88 |
| eyesmix_b1 | 7.95 | 84.86, 83.89, 84.30 |
| eyesmix_b2 | 8.95 | 85.50, 84.48, 84.45 |
| eyesmix_b3 | 9.95 | 85.83, 84.73, 84.55 |
| eyesmix_b4 | 10.95 | 85.80, 84.83, 84.53 |
| eyesmix_stop | 8.05 | 85.88, 84.83, 84.42 |
| glance | 6.95 | 82.11, 81.27, 82.44 |
| glance_noR | 6.95 | 70.99, 71.09, 64.70 |
| hires | 27.79 | 84.99, 85.55, 85.75 |
| motion_b2 | 8.95 | 84.78, 84.05, 85.06 |
| random_b1 | 7.95 | 82.75, 82.02, 83.12 |
| random_b2 | 8.95 | 83.19, 82.56, 83.50 |
| random_b3 | 9.95 | 83.66, 83.04, 83.78 |
| random_b4 | 10.95 | 83.87, 83.35, 83.95 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.481 |
| hires | 0.546 |
| random_b2 | 0.496 |
| motion_b2 | 0.521 |
| attention_b2 | 0.524 |
| eyes_b2 | 0.519 |
| eyes_stop | 0.517 |
| best_pair | 0.630 |
| all9 | 0.525 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +3.07 | 152 |
| 2-3 | +3.13 | 713 |
| 4-6 | +2.78 | 1080 |
| 7-10 | +2.55 | 1985 |

## dinov2-boxes-where  (3930 test clips, 3 seeds, rho=0.79, lam=0.10)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +3.88 pts [+2.36, +6.23], McNemar p<=1.68e-15 | >= +2 pts | PASS |
| 2. Beats random | +3.25 pts [+2.17, +4.87] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -4.68 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.981 (pooled 0.973); adaptive eyes 0.893; greedy finds the best pair in 43% of clips, falls below 3/4 in 1.9% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.16, MC 0.02, closed~MC 0.11 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.043 vs no-R 0.174; glance 0.016 vs 0.202; All-9 0.106 vs 0.293 | <= 0.05 | PASS |

Accuracy: glance 81.82, eyes(2) 85.69, random(2) 82.44, best pair (oracle) 92.00, All-9 77.13

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 85.42 | 27.8 | +0.27 [-0.66, +1.64] |
| B3 attention | 85.19 | 8.9 | +0.50 [-0.15, +1.36] |
| B4 motion | 85.53 | 8.9 | +0.16 [-0.59, +1.31] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 83.84, 72.04, 75.54 |
| all9_noR | 69.48 | 76.87, 60.69, 61.49 |
| attention_b2 | 8.95 | 85.52, 84.48, 85.59 |
| best_pair | 8.95 | 92.04, 91.73, 92.24 |
| eyes_b1 | 7.95 | 85.65, 85.57, 85.75 |
| eyes_b2 | 8.95 | 86.18, 85.14, 85.77 |
| eyes_b3 | 9.95 | 86.36, 83.77, 85.21 |
| eyes_b4 | 10.95 | 86.39, 82.49, 83.74 |
| eyes_mc_b2 | 8.95 | 85.57, 83.87, 85.16 |
| eyes_noR_b2 | 8.95 | 77.25, 76.16, 72.28 |
| eyes_overlapR_b2 | 8.95 | 86.62, 86.34, 86.71 |
| eyes_stop | 7.91 | 86.51, 84.20, 85.01 |
| glance | 6.95 | 82.01, 81.15, 82.31 |
| glance_noR | 6.95 | 70.99, 71.09, 64.70 |
| hires | 27.79 | 84.99, 85.55, 85.75 |
| motion_b2 | 8.95 | 85.73, 85.09, 85.80 |
| random_b1 | 7.95 | 82.58, 81.52, 82.68 |
| random_b2 | 8.95 | 82.87, 81.76, 82.72 |
| random_b3 | 9.95 | 83.32, 81.58, 82.65 |
| random_b4 | 10.95 | 83.53, 81.36, 82.43 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.481 |
| hires | 0.546 |
| random_b2 | 0.517 |
| motion_b2 | 0.571 |
| attention_b2 | 0.577 |
| eyes_b2 | 0.586 |
| eyes_stop | 0.599 |
| best_pair | 0.735 |
| all9 | 0.524 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +9.43 | 152 |
| 2-3 | +5.66 | 713 |
| 4-6 | +2.65 | 1080 |
| 7-10 | +3.48 | 1985 |

## dinov2-boxes-where-shaped  (3930 test clips, 3 seeds, rho=0.81, lam=0.53)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +2.69 pts [+1.67, +3.98], McNemar p<=4.49e-07 | >= +2 pts | PASS |
| 2. Beats random | +1.54 pts [+0.91, +2.32] | >= +1 pt | PASS |
| 3. Gain recovered | 0.83 (All-9 gains +3.24 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.973 (pooled 0.969); adaptive eyes 0.843; greedy finds the best pair in 66% of clips, falls below 3/4 in 1.1% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.29, MC 0.27, closed~MC 0.71 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.012 vs no-R 0.183; glance 0.017 vs 0.202; All-9 0.008 vs 0.195 | <= 0.05 | PASS |

Accuracy: glance 81.93, eyes(2) 84.62, random(2) 83.08, best pair (oracle) 89.09, All-9 85.17

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 85.42 | 27.8 | -0.80 [-1.51, -0.32] |
| B3 attention | 84.58 | 8.9 | +0.04 [-0.39, +0.46] |
| B4 motion | 84.62 | 8.9 | -0.00 [-0.45, +0.63] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 85.62, 85.14, 84.78 |
| all9_noR | 69.48 | 78.58, 81.17, 66.79 |
| attention_b2 | 8.95 | 84.58, 84.05, 85.14 |
| best_pair | 8.95 | 89.21, 88.58, 89.51 |
| eyes_b1 | 7.95 | 84.86, 83.69, 83.91 |
| eyes_b2 | 8.95 | 85.19, 84.58, 84.12 |
| eyes_b3 | 9.95 | 85.55, 84.73, 84.68 |
| eyes_b4 | 10.95 | 85.83, 84.99, 84.86 |
| eyes_mc_b2 | 8.95 | 85.34, 84.17, 84.53 |
| eyes_noR_b2 | 8.95 | 75.27, 75.45, 68.90 |
| eyes_overlapR_b2 | 8.95 | 86.21, 85.29, 86.03 |
| eyes_stop | 8.07 | 85.42, 84.94, 84.86 |
| glance | 6.95 | 82.11, 81.27, 82.44 |
| glance_noR | 6.95 | 70.99, 71.09, 64.70 |
| hires | 27.79 | 84.99, 85.55, 85.75 |
| motion_b2 | 8.95 | 84.78, 84.05, 85.06 |
| random_b1 | 7.95 | 82.75, 82.02, 83.12 |
| random_b2 | 8.95 | 83.19, 82.56, 83.50 |
| random_b3 | 9.95 | 83.66, 83.04, 83.78 |
| random_b4 | 10.95 | 83.87, 83.35, 83.95 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.481 |
| hires | 0.546 |
| random_b2 | 0.496 |
| motion_b2 | 0.521 |
| attention_b2 | 0.524 |
| eyes_b2 | 0.516 |
| eyes_stop | 0.519 |
| best_pair | 0.630 |
| all9 | 0.525 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +2.85 | 152 |
| 2-3 | +3.09 | 713 |
| 4-6 | +2.65 | 1080 |
| 7-10 | +2.55 | 1985 |

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

## dinov3-boxes-preview  (3930 test clips, 3 seeds, rho=0.79, lam=0.07)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +4.89 pts [+2.71, +7.83], McNemar p<=1.33e-20 | >= +2 pts | PASS |
| 2. Beats random | +3.97 pts [+2.84, +5.64] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -10.17 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.978 (pooled 0.973); adaptive eyes 0.893; greedy finds the best pair in 44% of clips, falls below 3/4 in 2.3% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.16, MC 0.07, closed~MC 0.15 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.037 vs no-R 0.235; glance 0.018 vs 0.257; All-9 0.142 vs 0.404 | <= 0.05 | PASS |

Accuracy: glance 78.36, eyes(2) 83.26, random(2) 79.29, best pair (oracle) 90.59, All-9 68.19

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 82.15 | 27.8 | +1.11 [-0.30, +3.10] |
| B3 attention | 85.39 | 8.9 | -2.14 [-2.76, -1.36] |
| B4 motion | 83.38 | 8.9 | -0.13 [-1.12, +1.40] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 79.69, 60.69, 64.21 |
| all9_noR | 69.48 | 68.52, 46.54, 49.35 |
| attention_b2 | 8.95 | 85.27, 84.78, 86.15 |
| best_pair | 8.95 | 90.33, 90.28, 91.19 |
| eyes_b1 | 7.95 | 83.69, 83.41, 82.97 |
| eyes_b2 | 8.95 | 84.02, 82.39, 83.38 |
| eyes_b3 | 9.95 | 83.89, 80.56, 82.06 |
| eyes_b4 | 10.95 | 83.69, 78.35, 80.05 |
| eyes_mc_b2 | 8.95 | 82.82, 81.68, 84.07 |
| eyes_noR_b2 | 8.95 | 69.87, 67.05, 65.23 |
| eyes_overlapR_b2 | 8.95 | 85.06, 83.05, 84.37 |
| eyes_stop | 8.14 | 84.15, 80.25, 81.85 |
| eyesmix_b1 | 7.95 | 82.98, 81.42, 82.39 |
| eyesmix_b2 | 8.95 | 83.33, 80.31, 82.18 |
| eyesmix_b3 | 9.95 | 82.82, 77.76, 80.33 |
| eyesmix_b4 | 10.95 | 83.05, 74.96, 78.57 |
| eyesmix_stop | 8.24 | 83.56, 78.14, 80.38 |
| glance | 6.95 | 78.58, 77.74, 78.80 |
| glance_noR | 6.95 | 61.48, 61.65, 57.90 |
| hires | 27.79 | 82.11, 81.45, 82.90 |
| motion_b2 | 8.95 | 83.46, 83.00, 83.71 |
| random_b1 | 7.95 | 79.13, 78.36, 79.47 |
| random_b2 | 8.95 | 79.70, 78.38, 79.81 |
| random_b3 | 9.95 | 80.44, 77.91, 79.73 |
| random_b4 | 10.95 | 80.51, 77.33, 79.47 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.402 |
| hires | 0.475 |
| random_b2 | 0.456 |
| motion_b2 | 0.519 |
| attention_b2 | 0.570 |
| eyes_b2 | 0.535 |
| eyes_stop | 0.543 |
| best_pair | 0.683 |
| all9 | 0.453 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +10.75 | 152 |
| 2-3 | +7.29 | 713 |
| 4-6 | +3.36 | 1080 |
| 7-10 | +4.42 | 1985 |

## dinov3-boxes-preview-shaped  (3930 test clips, 3 seeds, rho=0.82, lam=0.53)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +3.77 pts [+2.60, +5.14], McNemar p<=5.85e-16 | >= +2 pts | PASS |
| 2. Beats random | +2.17 pts [+1.51, +2.95] | >= +1 pt | PASS |
| 3. Gain recovered | 0.81 (All-9 gains +4.67 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.971 (pooled 0.972); adaptive eyes 0.839; greedy finds the best pair in 66% of clips, falls below 3/4 in 1.3% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.28, MC 0.29, closed~MC 0.70 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.018 vs no-R 0.236; glance 0.025 vs 0.257; All-9 0.017 vs 0.259 | <= 0.05 | PASS |

Accuracy: glance 78.63, eyes(2) 82.40, random(2) 80.23, best pair (oracle) 87.40, All-9 83.31

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 82.15 | 27.8 | +0.25 [-0.86, +1.17] |
| B3 attention | 83.64 | 8.9 | -1.24 [-2.05, -0.68] |
| B4 motion | 82.14 | 8.9 | +0.26 [-0.21, +0.86] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 83.74, 83.13, 83.07 |
| all9_noR | 69.48 | 70.41, 77.48, 56.17 |
| attention_b2 | 8.95 | 83.69, 83.33, 83.91 |
| best_pair | 8.95 | 87.33, 87.10, 87.78 |
| eyes_b1 | 7.95 | 82.34, 81.32, 80.83 |
| eyes_b2 | 8.95 | 82.85, 82.32, 82.06 |
| eyes_b3 | 9.95 | 83.26, 82.44, 82.49 |
| eyes_b4 | 10.95 | 83.21, 82.49, 82.74 |
| eyes_mc_b2 | 8.95 | 82.98, 81.58, 82.23 |
| eyes_noR_b2 | 8.95 | 65.67, 67.38, 62.23 |
| eyes_overlapR_b2 | 8.95 | 83.94, 83.00, 83.97 |
| eyes_stop | 8.31 | 83.00, 82.47, 82.67 |
| eyesmix_b1 | 7.95 | 81.98, 81.15, 81.17 |
| eyesmix_b2 | 8.95 | 83.05, 81.96, 81.95 |
| eyesmix_b3 | 9.95 | 83.10, 82.52, 82.01 |
| eyesmix_b4 | 10.95 | 83.28, 82.70, 82.36 |
| eyesmix_stop | 8.26 | 82.98, 82.70, 82.23 |
| glance | 6.95 | 78.75, 78.02, 79.16 |
| glance_noR | 6.95 | 61.48, 61.65, 57.90 |
| hires | 27.79 | 82.11, 81.45, 82.90 |
| motion_b2 | 8.95 | 82.09, 81.78, 82.57 |
| random_b1 | 7.95 | 79.59, 78.97, 80.07 |
| random_b2 | 8.95 | 80.15, 79.73, 80.81 |
| random_b3 | 9.95 | 80.80, 80.18, 81.32 |
| random_b4 | 10.95 | 81.16, 80.57, 81.70 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.402 |
| hires | 0.475 |
| random_b2 | 0.434 |
| motion_b2 | 0.460 |
| attention_b2 | 0.513 |
| eyes_b2 | 0.468 |
| eyes_stop | 0.458 |
| best_pair | 0.582 |
| all9 | 0.477 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +4.61 | 152 |
| 2-3 | +4.11 | 713 |
| 4-6 | +3.49 | 1080 |
| 7-10 | +3.73 | 1985 |

## dinov3-boxes-where-shaped  (3930 test clips, 3 seeds, rho=0.82, lam=0.53)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +3.80 pts [+2.64, +5.17], McNemar p<=2.59e-16 | >= +2 pts | PASS |
| 2. Beats random | +2.21 pts [+1.56, +2.97] | >= +1 pt | PASS |
| 3. Gain recovered | 0.81 (All-9 gains +4.67 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.971 (pooled 0.972); adaptive eyes 0.840; greedy finds the best pair in 65% of clips, falls below 3/4 in 1.2% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.29, MC 0.29, closed~MC 0.70 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.018 vs no-R 0.236; glance 0.025 vs 0.257; All-9 0.017 vs 0.259 | <= 0.05 | PASS |

Accuracy: glance 78.63, eyes(2) 82.43, random(2) 80.23, best pair (oracle) 87.40, All-9 83.31

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 82.15 | 27.8 | +0.29 [-0.82, +1.20] |
| B3 attention | 83.64 | 8.9 | -1.20 [-2.01, -0.66] |
| B4 motion | 82.14 | 8.9 | +0.30 [-0.16, +0.89] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 83.74, 83.13, 83.07 |
| all9_noR | 69.48 | 70.41, 77.48, 56.17 |
| attention_b2 | 8.95 | 83.69, 83.33, 83.91 |
| best_pair | 8.95 | 87.33, 87.10, 87.78 |
| eyes_b1 | 7.95 | 82.39, 81.25, 80.91 |
| eyes_b2 | 8.95 | 82.90, 82.34, 82.08 |
| eyes_b3 | 9.95 | 83.44, 82.54, 82.46 |
| eyes_b4 | 10.95 | 83.23, 82.42, 82.79 |
| eyes_mc_b2 | 8.95 | 82.90, 81.81, 82.23 |
| eyes_noR_b2 | 8.95 | 65.80, 67.43, 62.26 |
| eyes_overlapR_b2 | 8.95 | 83.84, 82.95, 84.04 |
| eyes_stop | 8.30 | 83.10, 82.42, 82.69 |
| glance | 6.95 | 78.75, 78.02, 79.16 |
| glance_noR | 6.95 | 61.48, 61.65, 57.90 |
| hires | 27.79 | 82.11, 81.45, 82.90 |
| motion_b2 | 8.95 | 82.09, 81.78, 82.57 |
| random_b1 | 7.95 | 79.59, 78.97, 80.07 |
| random_b2 | 8.95 | 80.15, 79.73, 80.81 |
| random_b3 | 9.95 | 80.80, 80.18, 81.32 |
| random_b4 | 10.95 | 81.16, 80.57, 81.70 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.402 |
| hires | 0.475 |
| random_b2 | 0.434 |
| motion_b2 | 0.460 |
| attention_b2 | 0.513 |
| eyes_b2 | 0.469 |
| eyes_stop | 0.460 |
| best_pair | 0.582 |
| all9 | 0.477 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +4.39 | 152 |
| 2-3 | +4.35 | 713 |
| 4-6 | +3.40 | 1080 |
| 7-10 | +3.78 | 1985 |

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

## siglip2-boxes  (3930 test clips, 3 seeds, rho=0.81, lam=0.03)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +7.12 pts [+4.78, +9.28], McNemar p<=4.71e-22 | >= +2 pts | PASS |
| 2. Beats random | +5.52 pts [+4.52, +6.86] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -19.08 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.980 (pooled 0.977); adaptive eyes 0.903; greedy finds the best pair in 42% of clips, falls below 3/4 in 1.5% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.13, MC 0.08, closed~MC 0.21 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.053 vs no-R 0.332; glance 0.012 vs 0.358; All-9 0.285 vs 0.542 | <= 0.05 | fail |

Accuracy: glance 69.19, eyes(2) 76.31, random(2) 70.79, best pair (oracle) 88.65, All-9 50.11

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 68.80 | 27.8 | +7.51 [+4.04, +11.89] |
| B4 motion | 78.30 | 8.9 | -1.98 [-2.94, -0.55] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 64.73, 34.78, 50.83 |
| all9_noR | 69.48 | 50.94, 23.44, 34.28 |
| best_pair | 8.95 | 88.68, 88.02, 89.28 |
| eyes_b1 | 7.95 | 76.84, 76.51, 76.61 |
| eyes_b2 | 8.95 | 77.38, 75.06, 76.51 |
| eyes_b3 | 9.95 | 76.90, 71.35, 74.98 |
| eyes_b4 | 10.95 | 75.73, 66.11, 72.49 |
| eyes_mc_b2 | 8.95 | 78.42, 74.40, 77.45 |
| eyes_noR_b2 | 8.95 | 56.77, 50.51, 55.84 |
| eyes_overlapR_b2 | 8.95 | 78.70, 76.62, 78.67 |
| eyes_stop | 8.66 | 77.05, 71.07, 74.65 |
| glance | 6.95 | 68.65, 69.08, 69.84 |
| glance_noR | 6.95 | 47.81, 45.39, 46.96 |
| hires | 27.79 | 68.98, 67.81, 69.64 |
| motion_b2 | 8.95 | 78.91, 77.25, 78.75 |
| random_b1 | 7.95 | 70.84, 69.72, 71.35 |
| random_b2 | 8.95 | 71.96, 68.85, 71.57 |
| random_b3 | 9.95 | 72.50, 67.90, 71.19 |
| random_b4 | 10.95 | 72.79, 66.43, 70.56 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.312 |
| hires | 0.279 |
| random_b2 | 0.367 |
| motion_b2 | 0.443 |
| eyes_b2 | 0.480 |
| eyes_stop | 0.461 |
| best_pair | 0.669 |
| all9 | 0.324 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +8.77 | 152 |
| 2-3 | +5.24 | 713 |
| 4-6 | +4.26 | 1080 |
| 7-10 | +9.24 | 1985 |

## siglip2-boxes-preview  (3930 test clips, 3 seeds, rho=0.81, lam=0.03)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +7.26 pts [+4.45, +9.91], McNemar p<=1.79e-16 | >= +2 pts | PASS |
| 2. Beats random | +5.66 pts [+4.38, +7.06] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -19.08 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.979 (pooled 0.976); adaptive eyes 0.906; greedy finds the best pair in 44% of clips, falls below 3/4 in 1.7% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.19, MC 0.09, closed~MC 0.22 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.061 vs no-R 0.331; glance 0.012 vs 0.358; All-9 0.285 vs 0.542 | <= 0.05 | fail |

Accuracy: glance 69.19, eyes(2) 76.45, random(2) 70.79, best pair (oracle) 88.65, All-9 50.11

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 68.80 | 27.8 | +7.64 [+4.53, +12.07] |
| B4 motion | 78.30 | 8.9 | -1.85 [-2.79, -0.51] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 64.73, 34.78, 50.83 |
| all9_noR | 69.48 | 50.94, 23.44, 34.28 |
| best_pair | 8.95 | 88.68, 88.02, 89.28 |
| eyes_b1 | 7.95 | 78.19, 76.54, 77.60 |
| eyes_b2 | 8.95 | 78.30, 74.20, 76.86 |
| eyes_b3 | 9.95 | 77.96, 69.13, 74.88 |
| eyes_b4 | 10.95 | 76.41, 62.67, 71.72 |
| eyes_mc_b2 | 8.95 | 77.25, 72.54, 76.41 |
| eyes_noR_b2 | 8.95 | 57.76, 50.74, 56.04 |
| eyes_overlapR_b2 | 8.95 | 79.57, 77.02, 79.31 |
| eyes_stop | 8.54 | 77.68, 69.57, 74.42 |
| eyesmix_b1 | 7.95 | 77.40, 74.66, 76.97 |
| eyesmix_b2 | 8.95 | 78.22, 74.89, 77.55 |
| eyesmix_b3 | 9.95 | 78.12, 71.27, 76.36 |
| eyesmix_b4 | 10.95 | 77.20, 67.33, 73.38 |
| eyesmix_stop | 8.74 | 77.61, 70.64, 75.21 |
| glance | 6.95 | 68.65, 69.08, 69.84 |
| glance_noR | 6.95 | 47.81, 45.39, 46.96 |
| hires | 27.79 | 68.98, 67.81, 69.64 |
| motion_b2 | 8.95 | 78.91, 77.25, 78.75 |
| random_b1 | 7.95 | 70.84, 69.72, 71.35 |
| random_b2 | 8.95 | 71.96, 68.85, 71.57 |
| random_b3 | 9.95 | 72.50, 67.90, 71.19 |
| random_b4 | 10.95 | 72.79, 66.43, 70.56 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.312 |
| hires | 0.279 |
| random_b2 | 0.367 |
| motion_b2 | 0.443 |
| eyes_b2 | 0.475 |
| eyes_stop | 0.461 |
| best_pair | 0.669 |
| all9 | 0.324 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +8.55 | 152 |
| 2-3 | +5.70 | 713 |
| 4-6 | +3.83 | 1080 |
| 7-10 | +9.59 | 1985 |

## siglip2-boxes-preview-shaped  (3930 test clips, 3 seeds, rho=0.83, lam=0.47)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +6.13 pts [+4.03, +8.30], McNemar p<=5.22e-30 | >= +2 pts | PASS |
| 2. Beats random | +2.89 pts [+2.11, +3.50] | >= +1 pt | PASS |
| 3. Gain recovered | 0.76 (All-9 gains +8.02 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.971 (pooled 0.971); adaptive eyes 0.835; greedy finds the best pair in 64% of clips, falls below 3/4 in 0.8% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.37, MC 0.31, closed~MC 0.71 | >= 0.3 | PASS |
| 6. Calibration (ECE) | eyes 0.016 vs no-R 0.332; glance 0.018 vs 0.358; All-9 0.024 vs 0.269 | <= 0.05 | PASS |

Accuracy: glance 69.69, eyes(2) 75.83, random(2) 72.94, best pair (oracle) 84.16, All-9 77.72

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 68.80 | 27.8 | +7.02 [+4.13, +10.60] |
| B4 motion | 76.36 | 8.9 | -0.53 [-1.36, +0.30] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 77.63, 76.87, 78.67 |
| all9_noR | 69.48 | 54.96, 69.95, 73.63 |
| best_pair | 8.95 | 84.48, 83.36, 84.68 |
| eyes_b1 | 7.95 | 75.47, 74.12, 73.94 |
| eyes_b2 | 8.95 | 76.87, 75.32, 75.31 |
| eyes_b3 | 9.95 | 77.48, 76.13, 76.00 |
| eyes_b4 | 10.95 | 77.79, 76.01, 75.97 |
| eyes_mc_b2 | 8.95 | 76.74, 75.24, 75.18 |
| eyes_noR_b2 | 8.95 | 52.82, 51.91, 54.26 |
| eyes_overlapR_b2 | 8.95 | 78.80, 77.38, 77.98 |
| eyes_stop | 9.08 | 77.74, 75.90, 76.00 |
| eyesmix_b1 | 7.95 | 75.52, 74.58, 75.08 |
| eyesmix_b2 | 8.95 | 77.38, 76.13, 77.20 |
| eyesmix_b3 | 9.95 | 77.56, 77.10, 77.68 |
| eyesmix_b4 | 10.95 | 78.07, 77.48, 78.09 |
| eyesmix_stop | 8.79 | 77.74, 77.48, 77.55 |
| glance | 6.95 | 69.39, 69.54, 70.17 |
| glance_noR | 6.95 | 47.81, 45.39, 46.96 |
| hires | 27.79 | 68.98, 67.81, 69.64 |
| motion_b2 | 8.95 | 76.77, 75.80, 76.53 |
| random_b1 | 7.95 | 71.82, 71.31, 71.89 |
| random_b2 | 8.95 | 73.20, 72.36, 73.26 |
| random_b3 | 9.95 | 74.25, 73.17, 73.93 |
| random_b4 | 10.95 | 74.78, 73.77, 74.48 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.311 |
| hires | 0.279 |
| random_b2 | 0.335 |
| motion_b2 | 0.381 |
| eyes_b2 | 0.373 |
| eyes_stop | 0.372 |
| best_pair | 0.518 |
| all9 | 0.378 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +1.97 | 152 |
| 2-3 | +5.19 | 713 |
| 4-6 | +4.85 | 1080 |
| 7-10 | +7.49 | 1985 |

## siglip2-boxes-where  (3930 test clips, 3 seeds, rho=0.81, lam=0.03)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +7.32 pts [+4.63, +9.80], McNemar p<=9.5e-15 | >= +2 pts | PASS |
| 2. Beats random | +5.72 pts [+4.53, +7.05] | >= +1 pt | PASS |
| 3. Gain recovered | nan (All-9 gains -19.08 pts) | >= 2/3 | fail |
| 4. Near-optimal | greedy/best 0.979 (pooled 0.977); adaptive eyes 0.908; greedy finds the best pair in 44% of clips, falls below 3/4 in 1.6% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.19, MC 0.09, closed~MC 0.22 | >= 0.3 | fail |
| 6. Calibration (ECE) | eyes 0.061 vs no-R 0.330; glance 0.012 vs 0.358; All-9 0.285 vs 0.542 | <= 0.05 | fail |

Accuracy: glance 69.19, eyes(2) 76.51, random(2) 70.79, best pair (oracle) 88.65, All-9 50.11

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 68.80 | 27.8 | +7.70 [+4.50, +12.09] |
| B4 motion | 78.30 | 8.9 | -1.79 [-2.67, -0.49] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 64.73, 34.78, 50.83 |
| all9_noR | 69.48 | 50.94, 23.44, 34.28 |
| best_pair | 8.95 | 88.68, 88.02, 89.28 |
| eyes_b1 | 7.95 | 78.27, 76.49, 77.65 |
| eyes_b2 | 8.95 | 78.50, 73.89, 77.14 |
| eyes_b3 | 9.95 | 78.12, 69.08, 75.08 |
| eyes_b4 | 10.95 | 76.77, 62.70, 71.98 |
| eyes_mc_b2 | 8.95 | 77.15, 72.32, 76.20 |
| eyes_noR_b2 | 8.95 | 58.22, 50.46, 55.99 |
| eyes_overlapR_b2 | 8.95 | 79.54, 76.64, 79.43 |
| eyes_stop | 8.54 | 77.99, 69.24, 74.37 |
| glance | 6.95 | 68.65, 69.08, 69.84 |
| glance_noR | 6.95 | 47.81, 45.39, 46.96 |
| hires | 27.79 | 68.98, 67.81, 69.64 |
| motion_b2 | 8.95 | 78.91, 77.25, 78.75 |
| random_b1 | 7.95 | 70.84, 69.72, 71.35 |
| random_b2 | 8.95 | 71.96, 68.85, 71.57 |
| random_b3 | 9.95 | 72.50, 67.90, 71.19 |
| random_b4 | 10.95 | 72.79, 66.43, 70.56 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.312 |
| hires | 0.279 |
| random_b2 | 0.367 |
| motion_b2 | 0.443 |
| eyes_b2 | 0.471 |
| eyes_stop | 0.461 |
| best_pair | 0.669 |
| all9 | 0.324 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +7.68 | 152 |
| 2-3 | +5.84 | 713 |
| 4-6 | +4.01 | 1080 |
| 7-10 | +9.62 | 1985 |

## siglip2-boxes-where-shaped  (3930 test clips, 3 seeds, rho=0.83, lam=0.47)

| Metric | Value | Threshold | |
|---|---|---|---|
| 1. Gain over glance | +6.27 pts [+4.25, +8.43], McNemar p<=3.72e-33 | >= +2 pts | PASS |
| 2. Beats random | +3.03 pts [+2.35, +3.59] | >= +1 pt | PASS |
| 3. Gain recovered | 0.78 (All-9 gains +8.02 pts) | >= 2/3 | PASS |
| 4. Near-optimal | greedy/best 0.970 (pooled 0.971); adaptive eyes 0.835; greedy finds the best pair in 64% of clips, falls below 3/4 in 0.7% | >= 3/4 | PASS |
| 5. Ranking (Spearman) | closed 0.38, MC 0.31, closed~MC 0.71 | >= 0.3 | PASS |
| 6. Calibration (ECE) | eyes 0.015 vs no-R 0.331; glance 0.018 vs 0.358; All-9 0.024 vs 0.269 | <= 0.05 | PASS |

Accuracy: glance 69.69, eyes(2) 75.96, random(2) 72.94, best pair (oracle) 84.16, All-9 77.72

| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |
|---|---|---|---|
| B1 high-res | 68.80 | 27.8 | +7.16 [+4.27, +10.76] |
| B4 motion | 76.36 | 8.9 | -0.40 [-1.15, +0.33] |

| method | passes/clip | accuracy per seed (%) |
|---|---|---|
| all9 | 69.48 | 77.63, 76.87, 78.67 |
| all9_noR | 69.48 | 54.96, 69.95, 73.63 |
| best_pair | 8.95 | 84.48, 83.36, 84.68 |
| eyes_b1 | 7.95 | 75.50, 74.15, 74.09 |
| eyes_b2 | 8.95 | 76.97, 75.37, 75.57 |
| eyes_b3 | 9.95 | 77.53, 75.83, 76.05 |
| eyes_b4 | 10.95 | 77.94, 75.80, 76.10 |
| eyes_mc_b2 | 8.95 | 76.95, 75.45, 75.03 |
| eyes_noR_b2 | 8.95 | 52.98, 51.83, 54.36 |
| eyes_overlapR_b2 | 8.95 | 78.91, 77.28, 78.04 |
| eyes_stop | 9.08 | 77.89, 75.75, 76.15 |
| glance | 6.95 | 69.39, 69.54, 70.17 |
| glance_noR | 6.95 | 47.81, 45.39, 46.96 |
| hires | 27.79 | 68.98, 67.81, 69.64 |
| motion_b2 | 8.95 | 76.77, 75.80, 76.53 |
| random_b1 | 7.95 | 71.82, 71.31, 71.89 |
| random_b2 | 8.95 | 73.20, 72.36, 73.26 |
| random_b3 | 9.95 | 74.25, 73.17, 73.93 |
| random_b4 | 10.95 | 74.78, 73.77, 74.48 |

Official metrics (mean over seeds):

| method | macro-F1 |
|---|---|
| glance | 0.311 |
| hires | 0.279 |
| random_b2 | 0.335 |
| motion_b2 | 0.381 |
| eyes_b2 | 0.374 |
| eyes_stop | 0.376 |
| best_pair | 0.518 |
| all9 | 0.378 |

Gain of eyes(2) over glance by clip length:

| frames T | gain (pts) | clips |
|---|---|---|
| 1 | +2.19 | 152 |
| 2-3 | +5.38 | 713 |
| 4-6 | +5.12 | 1080 |
| 7-10 | +7.52 | 1985 |
