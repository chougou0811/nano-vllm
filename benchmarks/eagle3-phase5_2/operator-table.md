# Rich Trace Operator Attribution

Rank0 GPU kernel service sums in ms, not serving wall fractions. Captured MLP has no individual Python leaf calls; its graph kernels are attributed to mlp_region. Zero in a leaf column does not mean that computation disappeared.

| mode | cell | MLP region | gate/up | down | attention | QKV | o_proj | LM head |
|---|---|---|---|---|---|---|---|---|
| eager | speculative-c1-long-short | 0.000 | 7.631 | 3.949 | 1.094 | 1.712 | 1.223 | 0.822 |
| eager | speculative-c1-mixed-output | 0.000 | 7.622 | 3.966 | 0.528 | 1.713 | 1.220 | 0.822 |
| eager | speculative-c2-long-short | 0.000 | 7.635 | 3.957 | 1.096 | 1.709 | 1.222 | 0.826 |
| eager | speculative-c2-mixed-output | 0.000 | 7.638 | 3.968 | 0.528 | 1.714 | 1.221 | 0.826 |
| eager | speculative-c4-long-short | 0.000 | 7.662 | 3.981 | 1.105 | 1.735 | 1.240 | 0.837 |
| eager | speculative-c4-mixed-output | 0.000 | 7.681 | 3.990 | 0.531 | 1.712 | 1.227 | 0.835 |
| mlp | speculative-c1-long-short | 11.697 | 0.000 | 0.000 | 1.095 | 1.724 | 1.228 | 0.823 |
| mlp | speculative-c1-mixed-output | 11.679 | 0.000 | 0.000 | 0.529 | 1.720 | 1.218 | 0.824 |
| mlp | speculative-c2-long-short | 11.746 | 0.000 | 0.000 | 1.095 | 1.733 | 1.229 | 0.830 |
| mlp | speculative-c2-mixed-output | 11.742 | 0.000 | 0.000 | 0.527 | 1.728 | 1.223 | 0.829 |
| mlp | speculative-c4-long-short | 11.850 | 0.000 | 0.000 | 1.103 | 1.746 | 1.244 | 0.837 |
| mlp | speculative-c4-mixed-output | 11.829 | 0.000 | 0.000 | 0.532 | 1.742 | 1.241 | 0.838 |
