# Accepted EAGLE-3 Numerical Contract

User acceptance recorded 2026-09-20 after Phase 1.1 audit:

**numerical semantics accepted under the documented BF16 cross-shape contract;
strict serial parity not guaranteed.**

- Historical strict serial-token parity failures remain failures.
- Speculation off preserves original behavior.
- Fixed-K greedy acceptance follows the actual parallel target verification logits.
- KV/state transaction invariants are exact requirements.
- Fixed inputs, shapes and execution mode must be deterministic.
- BF16 q=1 GEMV and q>1 GEMM need not produce universally identical tokens near ties.
- Every new disagreement still requires state, operator and high-precision
  diagnostics. No arbitrary raw-logit threshold grants acceptance.

See `benchmarks/eagle3-phase1.1/report.md` for the evidence and limitations.
This acceptance does not pre-approve new persistent-state numerical differences.
