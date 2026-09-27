# Runtime and Memory

Wall columns are seconds summed over trials, not per step. Generation = total draft wall minus serial conditioning, including full serving publication/packing overhead. Allocated/reserved are rank0 peak GiB; both ranks and per-trial numbers remain in JSON.

| process | c | mode | catch-up s | generation s | verify s | draft forwards | batched feedback forwards | effective feedback B | accepted/proposed | allocated GiB | reserved GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving-c1-tail | 1 | batch | 3.886 | 8.228 | 75.902 | 5657 | 0 | 1.000 | 0.4448 | 18.798 | 19.518 |
| serving-c1-tail | 1 | serial | 3.676 | 8.184 | 72.684 | 5657 | 0 | 1.000 | 0.4448 | 18.798 | 19.518 |
| serving-c1-tail-r2 | 1 | batch | 3.884 | 8.195 | 73.164 | 5657 | 0 | 1.000 | 0.4448 | 18.798 | 19.518 |
| serving-c1-tail-r2 | 1 | serial | 3.685 | 8.183 | 72.985 | 5657 | 0 | 1.000 | 0.4448 | 18.798 | 19.518 |
| serving-fresh-r3 | 1 | batch | 2.191 | 4.721 | 41.979 | 3273 | 0 | 1.000 | 0.6376 | 18.797 | 19.400 |
| serving-fresh-r3 | 1 | serial | 2.191 | 4.717 | 42.677 | 3273 | 0 | 1.000 | 0.6376 | 18.797 | 19.400 |
| serving-fresh-r3 | 2 | serial | 2.112 | 4.697 | 29.245 | 3282 | 0 | 1.000 | 0.6347 | 18.831 | 19.400 |
| serving-fresh-r3 | 2 | batch | 2.092 | 3.273 | 27.808 | 2537 | 745 | 1.520 | 0.6347 | 18.830 | 19.400 |
| serving-fresh-r3 | 4 | batch | 4.104 | 3.505 | 35.310 | 3673 | 1167 | 3.039 | 0.6256 | 18.879 | 19.400 |
| serving-fresh-r3 | 4 | serial | 4.153 | 9.471 | 40.395 | 6624 | 0 | 1.000 | 0.6253 | 18.880 | 19.400 |
| serving-main-r3 | 1 | serial | 12.251 | 26.623 | 253.780 | 18270 | 0 | 1.000 | 0.5215 | 18.797 | 19.492 |
| serving-main-r3 | 1 | batch | 12.388 | 26.611 | 258.729 | 18270 | 0 | 1.000 | 0.5215 | 18.797 | 19.492 |
| serving-main-r3 | 2 | batch | 11.361 | 17.146 | 152.775 | 13477 | 4680 | 1.636 | 0.5269 | 18.832 | 19.492 |
| serving-main-r3 | 2 | serial | 11.510 | 26.279 | 162.595 | 18154 | 0 | 1.000 | 0.5270 | 18.832 | 19.492 |
| serving-main-r3 | 4 | serial | 22.321 | 52.338 | 217.336 | 36404 | 0 | 1.000 | 0.5248 | 18.906 | 19.492 |
| serving-main-r3 | 4 | batch | 22.160 | 18.907 | 180.764 | 19956 | 6292 | 3.138 | 0.5248 | 18.904 | 19.492 |
