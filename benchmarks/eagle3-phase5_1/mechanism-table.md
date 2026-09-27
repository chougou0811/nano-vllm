# Mechanism Traces

Two profiled verification windows per row; rollback spans retained separately in raw analysis. Times in us. NCCL residency is not wire time. No trace outlier removed.

| trace | CPU launch calls | GPU kernels | NCCL kernels | copy time | graph launch CPU | GPU gap |
|---|---|---|---|---|---|---|
| b4-ctx256-layersNone-mlp-rank0.json | 523.00 | 643.00 | 84.00 | 57.35 | 1053.65 | 30967.83 |
| b4-ctx256-layersNone-eager-rank1.json | 632.00 | 632.00 | 84.00 | 4.37 | 0.00 | 25984.17 |
| b4-ctx768-layersNone-eager-rank0.json | 643.00 | 643.00 | 84.00 | 17.78 | 0.00 | 34300.14 |
| b2-ctx256-layersNone-mlp-rank1.json | 512.00 | 632.00 | 84.00 | 42.18 | 1019.99 | 3919.02 |
| b2-ctx256-layersNone-mlp-rank0.json | 521.00 | 641.00 | 84.00 | 52.24 | 1020.66 | 31633.88 |
| b2-ctx768-layersNone-eager-rank1.json | 632.00 | 632.00 | 84.00 | 4.27 | 0.00 | 3853.98 |
| b2-ctx768-layersNone-eager-rank0.json | 641.00 | 641.00 | 84.00 | 14.27 | 0.00 | 34985.77 |
| b2-ctx768-layersNone-mlp-rank1.json | 512.00 | 632.00 | 84.00 | 42.37 | 999.14 | 1861.87 |
| b1-ctx256-layersNone-eager-rank1.json | 632.00 | 632.00 | 84.00 | 4.26 | 0.00 | 29829.21 |
| b1-ctx256-layersNone-eager-rank0.json | 640.00 | 640.00 | 84.00 | 11.65 | 0.00 | 22742.04 |
| b2-ctx768-layersNone-mlp-rank0.json | 521.00 | 641.00 | 84.00 | 52.26 | 1036.21 | 27021.01 |
| b1-ctx256-layersNone-mlp-rank1.json | 512.00 | 632.00 | 84.00 | 42.34 | 953.31 | 10079.84 |
| b1-ctx256-layersNone-mlp-rank0.json | 520.00 | 640.00 | 84.00 | 47.91 | 1016.14 | 27971.46 |
| b4-ctx256-layersNone-eager-rank0.json | 643.00 | 643.00 | 84.00 | 17.78 | 0.00 | 25523.04 |
| b1-ctx768-layersNone-eager-rank1.json | 632.00 | 632.00 | 84.00 | 5.30 | 0.00 | 13999.01 |
| b1-ctx768-layersNone-eager-rank0.json | 640.00 | 640.00 | 84.00 | 10.80 | 0.00 | 33867.31 |
| b4-ctx768-layersNone-eager-rank1.json | 632.00 | 632.00 | 84.00 | 3.34 | 0.00 | 7353.99 |
| b1-ctx768-layersNone-mlp-rank1.json | 512.00 | 632.00 | 84.00 | 42.61 | 949.45 | 14289.12 |
| b1-ctx768-layersNone-mlp-rank0.json | 520.00 | 640.00 | 84.00 | 48.10 | 1017.90 | 19312.94 |
| b4-ctx256-layersNone-mlp-rank1.json | 512.00 | 632.00 | 84.00 | 43.02 | 1018.13 | 10986.71 |
| b2-ctx256-layersNone-eager-rank1.json | 632.00 | 632.00 | 84.00 | 5.23 | 0.00 | 11990.40 |
| b2-ctx256-layersNone-eager-rank0.json | 641.00 | 641.00 | 84.00 | 14.30 | 0.00 | 34446.69 |
| b4-ctx768-layersNone-mlp-rank1.json | 512.00 | 632.00 | 84.00 | 43.11 | 1093.86 | 6149.97 |
| b4-ctx768-layersNone-mlp-rank0.json | 523.00 | 643.00 | 84.00 | 57.04 | 1003.06 | 43891.96 |
