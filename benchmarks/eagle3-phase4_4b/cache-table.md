# Exact-Key Cache

Hit rate denominator is all verification batches, not just eligible batches.

| System | c | Workload | Eligible | Hit | Unique keys/trial mean | Captures | Replays | Evictions | Capture ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| never | 1 | short-short | 90.4% | 0.0% | 26.2 | 0 | 0 | 0 | 0.0 |
| never | 1 | short-long | 97.9% | 0.0% | 121.2 | 0 | 0 | 0 | 0.0 |
| never | 1 | long-short | 91.2% | 0.0% | 25.0 | 0 | 0 | 0 | 0.0 |
| never | 1 | long-long | 97.8% | 0.0% | 100.0 | 0 | 0 | 0 | 0.0 |
| never | 1 | mixed-prompt | 93.8% | 0.0% | 61.0 | 0 | 0 | 0 | 0.0 |
| never | 1 | mixed-output | 95.8% | 0.0% | 45.8 | 0 | 0 | 0 | 0.0 |
| never | 2 | short-short | 83.3% | 0.0% | 14.8 | 0 | 0 | 0 | 0.0 |
| never | 2 | short-long | 96.3% | 0.0% | 71.0 | 0 | 0 | 0 | 0.0 |
| never | 2 | long-short | 86.2% | 0.0% | 13.4 | 0 | 0 | 0 | 0.0 |
| never | 2 | long-long | 95.6% | 0.0% | 62.0 | 0 | 0 | 0 | 0.0 |
| never | 2 | mixed-prompt | 90.2% | 0.0% | 29.6 | 0 | 0 | 0 | 0.0 |
| never | 2 | mixed-output | 93.7% | 0.0% | 37.2 | 0 | 0 | 0 | 0.0 |
| never | 4 | short-short | 80.2% | 0.0% | 13.0 | 0 | 0 | 0 | 0.0 |
| never | 4 | short-long | 91.7% | 0.0% | 66.4 | 0 | 0 | 0 | 0.0 |
| never | 4 | long-short | 76.4% | 0.0% | 9.2 | 0 | 0 | 0 | 0.0 |
| never | 4 | long-long | 91.5% | 0.0% | 54.6 | 0 | 0 | 0 | 0.0 |
| never | 4 | mixed-prompt | 87.9% | 0.0% | 17.6 | 0 | 0 | 0 | 0.0 |
| never | 4 | mixed-output | 74.4% | 0.0% | 36.6 | 0 | 0 | 0 | 0.0 |
| second4 | 1 | short-short | 90.4% | 5.1% | 26.2 | 38 | 10 | 18 | 10313.2 |
| second4 | 1 | short-long | 97.9% | 0.3% | 121.2 | 80 | 2 | 60 | 21809.5 |
| second4 | 1 | long-short | 91.2% | 9.4% | 25.0 | 14 | 16 | 0 | 4062.9 |
| second4 | 1 | long-long | 97.8% | 1.4% | 100.0 | 80 | 10 | 60 | 21489.8 |
| second4 | 1 | mixed-prompt | 93.8% | 0.0% | 61.0 | 0 | 0 | 0 | 0.0 |
| second4 | 1 | mixed-output | 95.8% | 5.8% | 45.8 | 48 | 18 | 28 | 12970.1 |
| second4 | 2 | short-short | 83.3% | 0.0% | 14.8 | 11 | 0 | 1 | 3092.3 |
| second4 | 2 | short-long | 96.3% | 0.0% | 71.0 | 12 | 0 | 1 | 3707.1 |
| second4 | 2 | long-short | 86.2% | 0.0% | 13.4 | 8 | 0 | 0 | 2168.5 |
| second4 | 2 | long-long | 95.6% | 0.0% | 62.0 | 38 | 0 | 18 | 10096.3 |
| second4 | 2 | mixed-prompt | 90.2% | 0.0% | 29.6 | 0 | 0 | 0 | 0.0 |
| second4 | 2 | mixed-output | 93.7% | 0.0% | 37.2 | 6 | 0 | 0 | 1779.8 |
| second4 | 4 | short-short | 80.2% | 0.0% | 13.0 | 16 | 0 | 0 | 5167.0 |
| second4 | 4 | short-long | 91.7% | 0.0% | 66.4 | 28 | 0 | 16 | 9131.5 |
| second4 | 4 | long-short | 76.4% | 0.0% | 9.2 | 22 | 0 | 6 | 7077.1 |
| second4 | 4 | long-long | 91.5% | 0.0% | 54.6 | 58 | 0 | 38 | 18350.7 |
| second4 | 4 | mixed-prompt | 87.9% | 0.0% | 17.6 | 57 | 0 | 40 | 18580.0 |
| second4 | 4 | mixed-output | 74.4% | 0.0% | 36.6 | 0 | 0 | 0 | 0.0 |
| second2 | 1 | long-long | 97.8% | 0.4% | 100.0 | 80 | 3 | 70 | 21340.0 |
| second2 | 1 | mixed-output | 95.8% | 1.0% | 45.8 | 63 | 3 | 53 | 16675.8 |
| second2 | 2 | long-long | 95.6% | 0.0% | 62.0 | 38 | 0 | 28 | 10684.5 |
| second2 | 2 | mixed-output | 93.7% | 0.0% | 37.2 | 6 | 0 | 1 | 1670.3 |
| second2 | 4 | long-long | 91.5% | 0.0% | 54.6 | 58 | 0 | 48 | 18240.3 |
| second2 | 4 | mixed-output | 74.4% | 0.0% | 36.6 | 0 | 0 | 0 | 0.0 |
| second8 | 1 | long-long | 97.8% | 1.7% | 100.0 | 80 | 12 | 40 | 21299.3 |
| second8 | 1 | mixed-output | 95.8% | 9.7% | 45.8 | 36 | 30 | 0 | 9819.4 |
| second8 | 2 | long-long | 95.6% | 0.0% | 62.0 | 38 | 0 | 9 | 10610.5 |
| second8 | 2 | mixed-output | 93.7% | 0.0% | 37.2 | 6 | 0 | 0 | 1688.8 |
| second8 | 4 | long-long | 91.5% | 0.0% | 54.6 | 58 | 0 | 22 | 18512.8 |
| second8 | 4 | mixed-output | 74.4% | 0.0% | 36.6 | 0 | 0 | 0 | 0.0 |
