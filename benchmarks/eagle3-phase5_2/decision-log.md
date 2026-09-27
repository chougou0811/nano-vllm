# Phase 5.2 Decision Log

## D0: Before new GPU measurements

- Scope: profiling, external source audit and design only. No new inference
  feature, historical edits or Git commit. Phase5.1 remains opt-in experimental.
- Measure both literal eager and frozen MLP replay in separate fresh processes.
  c1/2/4; seven frozen Phase5.1 workload families; three new unprofiled repeats
  per cell (126 trials total). Rotate workload order and reverse concurrency
  order between modes. All samples retained. This is characterization, not a
  new adoption claim or independent process replication per repeat.
- Finish wall measurements before profiling. Separate event-only, minimal and
  rich traces at two representative families plus short full closed-loop traces
  including prefill/replacement/fallback. Check output parity and cleanup.
- Report metadata/RPC timings as nested components, not additive independent
  wall fractions. Draft catch-up is a subset of proposal. NCCL residency and
  rank skew are not pure transport time. Graph kernels need correlation to
  graph launch, not assumptions from Python module invocation counts.
- External selection follows current measurements. No candidate is selected
  yet; do not assume graph, drafting, attention or communication is the answer.
- Prefer an isolated transfer with preserved state semantics and checkpoint;
  require hardware evidence and an explicit falsifiable next-phase MVP.

## D1: Offline parser correction

- Evidence: clipped remaining-budget decode can have no draft call and therefore
  no `draft_ns` key. The first summary attempt raised KeyError.
- Decision: represent missing draft timing for those steps as0, retaining all
  samples. No production or historical instrumentation change, no GPU rerun.
- Alternative rejected: discard those decode steps or infer a draft call.
- Impact: correct denominator/zero-cost handling; no performance filtering.

## D2: Selection after all new runs and trace analysis

- Evidence:126 trials,63 exact paired outputs/statistics,60 traces. Target remains
  largest at76.08/67.49/60.28%; draft grows13.39/19.29/24.22%, with mean draft
  step6.10/11.16/21.44ms. MLP remains about70% of non-NCCL kernel service;
  explicit metadata is small, driver P2P access false both ways.
- Decision: Primary is across-request EAGLE draft-step batching using the current
  checkpoint; first MVP preserves serial conditioning. Secondary is bounded
  BF16 MLP tactic selection, not another graph policy.
- Alternatives: broader graph/GPU-native target execution has opportunity but
  larger runtime/ownership scope; metadata/attention are too small here; custom
  peer collectives fail a current hardware prerequisite; replacing the drafter
  adds checkpoint/features/memory uncertainty before testing execution batching.
- Why this Primary despite target being largest: risk-adjusted bounded next
  experiment for c2/c4, not a claim that draft dominates. Backup directly targets
  MLP but has unproven Ada kernel advantage. Neither feature is implemented.
- Impact: keep Phase5.1 experimental. New c2 speedup1.053 differs from historical
  1.019; do not force a2% narrative or retroactively change5.1's adoption result.
  No benchmark outlier was removed. Await approval for the next implementation.
