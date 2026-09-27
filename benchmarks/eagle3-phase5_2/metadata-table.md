# Event-only Host Diagnostics

Rank0 ms; nested within target wall time. prepare_prefill includes tensor construction/copies, not pure CPU loops. _status includes synchronization. Dispatch is shared-memory packing/publication; rank1 read waits are not counted as extra serving latency.

| mode | cell | layout | descriptors | prepare input | status | dispatch |
|---|---|---|---|---|---|---|
| eager | speculative-c1-long-short | 0.0126 | 0.0106 | 0.1419 | 0.2505 | 0.0420 |
| eager | speculative-c1-mixed-output | 0.0136 | 0.0084 | 0.1233 | 0.2012 | 0.0426 |
| eager | speculative-c2-long-short | 0.0170 | 0.0166 | 0.1438 | 0.2326 | 0.0424 |
| eager | speculative-c2-mixed-output | 0.0161 | 0.0120 | 0.1450 | 0.2278 | 0.0445 |
| eager | speculative-c4-long-short | 0.0209 | 0.0259 | 0.1503 | 0.2530 | 0.0463 |
| eager | speculative-c4-mixed-output | 0.0216 | 0.0167 | 0.1526 | 0.2348 | 0.0491 |
| mlp | speculative-c1-long-short | 0.0139 | 0.0131 | 0.1545 | 0.2377 | 0.0464 |
| mlp | speculative-c1-mixed-output | 0.0135 | 0.0100 | 0.1578 | 0.2321 | 0.0495 |
| mlp | speculative-c2-long-short | 0.0178 | 0.0170 | 0.1597 | 0.2400 | 0.0484 |
| mlp | speculative-c2-mixed-output | 0.0171 | 0.0121 | 0.1580 | 0.2472 | 0.0506 |
| mlp | speculative-c4-long-short | 0.0226 | 0.0309 | 0.1723 | 0.2690 | 0.0478 |
| mlp | speculative-c4-mixed-output | 0.0203 | 0.0163 | 0.1482 | 0.2199 | 0.0488 |
