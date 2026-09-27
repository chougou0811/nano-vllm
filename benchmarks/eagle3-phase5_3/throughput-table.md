# Serving Paired Results

Geomean serial wall / batch wall. Tail ratios use the median of each repeat's percentile. All repeats included. Incomplete/error processes remain in summary.json but are not presented as completed five-repeat cells.

| process | family | c | repeats | speedup | improving | P95 ITL ratio | P99 ITL ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| serving-c1-tail | long-long | 1 | 5 | 0.9530 | 1 | 1.0193 | 1.0820 |
| serving-c1-tail | copy-pattern | 1 | 5 | 0.9704 | 1 | 1.0337 | 1.0445 |
| serving-c1-tail-r2 | long-long | 1 | 5 | 0.9857 | 4 | 0.9906 | 0.9704 |
| serving-c1-tail-r2 | copy-pattern | 1 | 5 | 1.0027 | 2 | 0.9907 | 1.0325 |
| serving-fresh-r3 | heldout-mixed | 1 | 5 | 1.0016 | 2 | 1.0004 | 0.9936 |
| serving-fresh-r3 | heldout-mixed | 2 | 5 | 1.0641 | 4 | 0.7829 | 0.9775 |
| serving-fresh-r3 | heldout-mixed | 4 | 5 | 1.1571 | 5 | 0.8145 | 0.9292 |
| serving-fresh-r3 | routing-table | 1 | 5 | 1.0176 | 3 | 0.9666 | 0.9766 |
| serving-fresh-r3 | routing-table | 2 | 5 | 1.0762 | 5 | 0.8261 | 0.8915 |
| serving-fresh-r3 | routing-table | 4 | 5 | 1.2545 | 5 | 0.7860 | 0.9696 |
| serving-main-r3 | short-short | 1 | 5 | 1.0606 | 3 | 0.9881 | 0.9957 |
| serving-main-r3 | short-short | 2 | 5 | 1.1333 | 5 | 0.7907 | 0.9625 |
| serving-main-r3 | short-short | 4 | 5 | 1.1754 | 5 | 0.7721 | 0.9468 |
| serving-main-r3 | short-long | 1 | 5 | 0.9672 | 3 | 0.9966 | 1.0008 |
| serving-main-r3 | short-long | 2 | 5 | 1.1685 | 5 | 0.8190 | 0.9491 |
| serving-main-r3 | short-long | 4 | 5 | 1.2617 | 5 | 0.7891 | 1.1630 |
| serving-main-r3 | long-short | 1 | 5 | 1.0349 | 3 | 0.9941 | 1.0056 |
| serving-main-r3 | long-short | 2 | 5 | 1.0858 | 5 | 0.8247 | 0.9942 |
| serving-main-r3 | long-short | 4 | 5 | 1.1388 | 5 | 0.8091 | 0.9792 |
| serving-main-r3 | long-long | 1 | 5 | 0.9416 | 2 | 1.3120 | 1.2463 |
| serving-main-r3 | long-long | 2 | 5 | 1.2492 | 5 | 0.7221 | 0.8812 |
| serving-main-r3 | long-long | 4 | 5 | 1.2488 | 5 | 0.8188 | 0.9422 |
| serving-main-r3 | mixed-prompt | 1 | 5 | 0.9955 | 3 | 1.0205 | 1.0158 |
| serving-main-r3 | mixed-prompt | 2 | 5 | 1.0551 | 5 | 0.9469 | 0.9491 |
| serving-main-r3 | mixed-prompt | 4 | 5 | 1.1794 | 5 | 0.8067 | 0.8157 |
| serving-main-r3 | mixed-output | 1 | 5 | 1.0051 | 4 | 1.0024 | 1.0119 |
| serving-main-r3 | mixed-output | 2 | 5 | 1.0273 | 5 | 0.9519 | 0.9539 |
| serving-main-r3 | mixed-output | 4 | 5 | 1.0889 | 5 | 0.8155 | 0.9179 |
| serving-main-r3 | heldout-mixed | 1 | 5 | 1.0373 | 3 | 0.9931 | 0.9128 |
| serving-main-r3 | heldout-mixed | 2 | 5 | 1.0337 | 4 | 0.8628 | 0.9629 |
| serving-main-r3 | heldout-mixed | 4 | 5 | 1.1193 | 5 | 0.8075 | 0.9292 |
| serving-main-r3 | copy-pattern | 1 | 5 | 0.9660 | 1 | 1.2296 | 1.1802 |
| serving-main-r3 | copy-pattern | 2 | 5 | 1.0854 | 5 | 0.8295 | 0.9487 |
| serving-main-r3 | copy-pattern | 4 | 5 | 1.3714 | 5 | 0.7225 | 0.9096 |
| serving-main-r3 | observatory-prose | 1 | 5 | 0.9999 | 3 | 1.0097 | 0.9854 |
| serving-main-r3 | observatory-prose | 2 | 5 | 1.0739 | 5 | 0.8472 | 0.8388 |
| serving-main-r3 | observatory-prose | 4 | 5 | 1.3701 | 5 | 0.7654 | 0.6344 |
| serving-main-r3 | routing-table | 1 | 5 | 0.9709 | 3 | 0.9978 | 1.1702 |
| serving-main-r3 | routing-table | 2 | 5 | 1.0508 | 3 | 0.9192 | 0.9616 |
| serving-main-r3 | routing-table | 4 | 5 | 1.2996 | 5 | 0.7050 | 0.9112 |
