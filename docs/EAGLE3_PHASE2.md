# Phase 2: Persistent Draft State

Accepted and frozen by the user on 2026-09-21. Phase 3 may select K but must
preserve the draft/target commit and rollback semantics documented here.

Phase 1 / 1.1 checkpoint: `087cc27`. Numerical semantics follow
`EAGLE3_NUMERICAL_CONTRACT.md`; historical serial parity failures remain failures.

## Ownership and Cursors

Each exclusive EAGLE request owns a `DraftState` on rank 0. Target KV remains
TP-sharded with the Phase 1 transaction protocol unchanged. Scheduler is frozen.

Let C be valid target KV rows and D the draft conditioned-prefix cursor.
Before a nonterminal proposal, target tokens have length C+1; the final token
is pending and has no target KV yet. Target features have length C.
Draft row i uses target feature i and target token i+1. Thus its final row may
condition on the pending token, without claiming that token has target KV.

1. Catch up draft rows [D,C) using verified target features and shifted tokens.
2. Save this confirmed cache separately. Generate K proposals using K-1
   autoregressive draft feedback rows in private, tentative cache tensors.
3. Discard all autoregressive feedback rows, including those whose token IDs
   will be accepted: their draft hidden inputs are not verified target features.
4. Target verifies pending token plus K proposals in parallel. Accept a tokens,
   keep 1+a target rows, zero rejected target suffix, and append accepted IDs
   plus fallback/bonus unless terminated. D stays C_old; C becomes C_old+1+a.
5. Next proposal incrementally catches up 1+a rows. Never reuse rejected or
   proposal-hidden-conditioned KV. On terminal EOS, tokens may have length C;
   there is no pending fallback, and no further proposal is run.

The pinned reference attention uses `torch.cat`, not in-place cache mutation.
Keeping the confirmed tuple separately avoids a sliced view retaining tentative
storage. This is semantic persistence, not a preallocated cache optimization:
each append still copies prefix KV in reference code. Initial prefill still
processes the complete prefix. Target feature history remains full-length.

## Modes and Measurement

`generate_eagle3(..., draft_state_mode="persistent")` opts in.
`full_rebuild` remains default and processes all C feature rows every proposal.
The Phase 1 reference `ReferenceDraft.propose` is retained unchanged.

Per-step metrics distinguish conditioning time, total proposal/verification
latency, processed rows, reused prefix rows, discarded feedback rows and cache
bytes. `rollback_tokens` counts evaluated tentative feedback rows (K-1), not
rejected proposal IDs. `discarded` retains the latter meaning.

Draft close runs in the session finally block, including intentional exceptions;
target close and page release retain the Phase 1 protocol. No state is shared
between requests. Request identity and shifted-prefix consistency are checked.

## Validation Status

The scoped implementation passed 72 unit tests and two TP=2 GPU audit runs.
See `benchmarks/eagle3-phase2/report.md` and `summary.json` for evidence.
Fixed-proposal replay separates exact target transaction equivalence from
full-vs-incremental BF16 draft shape differences. Raw draft KV is not bitwise
identical; 41 shadow proposals agreed, with FC shape and FP32 controls recorded.
This is bounded validation, not a universal numerical equivalence claim.
No Adaptive K or concurrent speculative serving was implemented.
