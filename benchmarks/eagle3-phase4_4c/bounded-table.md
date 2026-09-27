# Bounded validation (measured)

Speedup = paired eager wall time / graph wall time. Final cache clear is outside serving time.

| C | Family | Repeat | Eager s | Graph s | Speedup | Hits | Captures | Profitable lifetimes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | short-long | 0 | 895.428 | 980.877 | 0.9129 | 653 | 16 | 4 |
| 1 | short-long | 1 | 898.724 | 1012.426 | 0.8877 | 653 | 16 | 4 |
| 1 | short-long | 2 | 892.337 | 987.564 | 0.9036 | 653 | 16 | 4 |
| 2 | long-long | 0 | 583.932 | 730.046 | 0.7999 | 278 | 16 | 4 |
| 2 | long-long | 1 | 589.778 | 734.200 | 0.8033 | 278 | 16 | 4 |
| 2 | long-long | 2 | 593.891 | 732.132 | 0.8112 | 278 | 16 | 4 |
| 4 | long-long | 0 | 448.969 | 505.254 | 0.8886 | 47 | 16 | 3 |
| 4 | long-long | 1 | 454.905 | 508.463 | 0.8947 | 47 | 16 | 3 |
| 4 | long-long | 2 | 450.316 | 509.312 | 0.8842 | 47 | 16 | 3 |
