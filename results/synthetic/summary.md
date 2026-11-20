# Synthetic study (known R, 3 seeds x 1000 test clips)

Shrinkage lam chosen by held-out NLL per seed: [0.4, 0.5, 0.4]

## Accuracy / ECE against shrinkage of the within-frame table A

| lam | glance acc | glance ECE | eyes(2) acc | eyes(2) ECE | All-9 acc | All-9 ECE |
|---|---|---|---|---|---|---|
| 0.0 | 37.0 | 0.034 | 68.1 | 0.043 | 44.9 | 0.551 |
| 0.1 | 37.0 | 0.034 | 70.0 | 0.038 | 63.0 | 0.178 |
| 0.2 | 37.0 | 0.034 | 71.4 | 0.034 | 74.3 | 0.039 |
| 0.3 | 37.0 | 0.034 | 72.9 | 0.034 | 78.7 | 0.065 |
| 0.4 | 37.0 | 0.034 | 73.9 | 0.031 | 80.2 | 0.085 |
| 0.5 | 37.0 | 0.034 | 75.2 | 0.022 | 80.2 | 0.076 |
| 0.6 | 37.0 | 0.034 | 76.2 | 0.033 | 78.6 | 0.048 |
| 0.8 | 37.0 | 0.034 | 77.0 | 0.042 | 75.8 | 0.044 |
| 1.0 | 37.0 | 0.034 | 76.0 | 0.071 | 72.9 | 0.122 |

## Looks budget (lam chosen by held-out NLL)

| looks | passes | eyes acc | random acc |
|---|---|---|---|
| 0 | 5.5 | 37.0 ± 1.4 | 37.0 ± 1.4 |
| 1 | 6.5 | 67.6 ± 0.5 | 39.4 ± 1.2 |
| 2 | 7.5 | 74.1 ± 1.4 | 43.8 ± 1.8 |
| 3 | 8.5 | 75.6 ± 1.4 | 47.1 ± 0.6 |
| 4 | 9.5 | 76.0 ± 1.7 | 49.1 ± 0.6 |

All-9: 80.1% (55.1 passes); eyes(2) recovers 0.86 of All-9's gain over the glance.
Eyes without R: acc 67.1, ECE 0.195 (with R: ECE 0.025). All-9 without R: ECE 0.235.
