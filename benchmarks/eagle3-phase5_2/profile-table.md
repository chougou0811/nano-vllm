# Minimal Trace Diagnostics

Times in ms. Two verification windows/row. Profiler timings are perturbed; NCCL residence is not transport time. Rank comparison columns are shared across the paired ranks.

| mode | cell | rank | host | GPU envelope | GPU idle gap | NCCL residence | launch APIs | graph launch CPU | mean rank skew | joint NCCL residence |
|---|---|---|---|---|---|---|---|---|---|---|
| eager | speculative-c1-long-short | 0 | 71.413 | 71.030 | 1.991 | 52.019 | 640 | 0.000 | 0.606 | 1.170 |
| eager | speculative-c1-long-short | 1 | 71.329 | 70.825 | 52.547 | 1.315 | 632 | 0.000 | 0.606 | 1.170 |
| eager | speculative-c1-mixed-output | 0 | 72.236 | 71.816 | 2.016 | 53.343 | 640 | 0.000 | 0.621 | 1.205 |
| eager | speculative-c1-mixed-output | 1 | 72.164 | 71.576 | 53.706 | 1.338 | 632 | 0.000 | 0.621 | 1.205 |
| eager | speculative-c2-long-short | 0 | 72.890 | 72.494 | 1.905 | 53.535 | 641 | 0.000 | 0.619 | 1.572 |
| eager | speculative-c2-long-short | 1 | 72.847 | 72.261 | 53.413 | 1.823 | 632 | 0.000 | 0.619 | 1.572 |
| eager | speculative-c2-mixed-output | 0 | 81.531 | 81.148 | 1.684 | 62.960 | 641 | 0.000 | 0.732 | 1.584 |
| eager | speculative-c2-mixed-output | 1 | 81.419 | 80.805 | 62.480 | 1.877 | 632 | 0.000 | 0.732 | 1.584 |
| eager | speculative-c4-long-short | 0 | 74.015 | 73.591 | 1.712 | 54.698 | 643 | 0.000 | 0.616 | 3.042 |
| eager | speculative-c4-long-short | 1 | 73.899 | 73.207 | 52.841 | 3.216 | 632 | 0.000 | 0.616 | 3.042 |
| eager | speculative-c4-mixed-output | 0 | 76.316 | 75.884 | 2.522 | 56.750 | 643 | 0.000 | 0.641 | 3.041 |
| eager | speculative-c4-mixed-output | 1 | 76.134 | 75.459 | 55.579 | 3.279 | 632 | 0.000 | 0.641 | 3.041 |
| mlp | speculative-c1-long-short | 0 | 64.972 | 64.368 | 2.519 | 44.721 | 520 | 1.172 | 0.519 | 1.181 |
| mlp | speculative-c1-long-short | 1 | 64.982 | 64.148 | 45.751 | 1.298 | 512 | 1.218 | 0.519 | 1.181 |
| mlp | speculative-c1-mixed-output | 0 | 68.100 | 67.500 | 2.638 | 48.325 | 520 | 1.172 | 0.561 | 1.199 |
| mlp | speculative-c1-mixed-output | 1 | 68.087 | 67.248 | 49.240 | 1.340 | 512 | 1.255 | 0.561 | 1.199 |
| mlp | speculative-c2-long-short | 0 | 67.912 | 67.293 | 2.553 | 47.522 | 521 | 1.161 | 0.548 | 1.577 |
| mlp | speculative-c2-long-short | 1 | 67.901 | 66.956 | 47.956 | 1.836 | 512 | 1.250 | 0.548 | 1.577 |
| mlp | speculative-c2-mixed-output | 0 | 65.966 | 65.367 | 2.575 | 46.161 | 521 | 1.156 | 0.532 | 1.556 |
| mlp | speculative-c2-mixed-output | 1 | 65.971 | 65.117 | 46.667 | 1.833 | 512 | 1.237 | 0.532 | 1.556 |
| mlp | speculative-c4-long-short | 0 | 58.475 | 57.801 | 14.581 | 25.834 | 523 | 1.083 | 0.388 | 3.079 |
| mlp | speculative-c4-long-short | 1 | 58.234 | 57.616 | 27.278 | 13.032 | 512 | 1.122 | 0.388 | 3.079 |
| mlp | speculative-c4-mixed-output | 0 | 77.886 | 77.188 | 3.621 | 56.785 | 523 | 1.166 | 0.642 | 3.061 |
| mlp | speculative-c4-mixed-output | 1 | 77.717 | 77.048 | 56.952 | 3.339 | 512 | 1.369 | 0.642 | 3.061 |
