# Candidate Decision Matrix

This is an ordinal engineering decision, not a fabricated numeric ranking.
Full per-candidate requirements and evidence are in [candidate-pool.md](candidate-pool.md).
`Conditional` means source-compatible design requiring tests, not verified support.
Only C1 and C2 are selected. C2 is a fallback with an attribution gate.

| ID | Candidate | Local evidence match | 2x4090 / TP2 / Qwen14B | Train / checkpoint | Reuse | Change/risk | Bounded MVP | Decision |
|---|---|---|---|---|---|---|---|---|
| C1 | Context-independent target region replay | Strong launch mechanism; proposed serving gain unknown | Conditional / conditional / same math | No / no | Very high | Medium-high; graph storage and NCCL | Yes, one region type | **Primary** |
| C2 | Incremental input metadata | Rebuild visible; time attribution missing | Plausible / same RPC / same inputs | No / no | Very high | Medium; stale ownership and copies | Yes, one prep stage | **Secondary only if gate passes** |
| C3 | Whole async runtime | Gaps known, overlap slack unknown | Possible / difficult / same model | No / no | Medium | High; multi-step ownership | Not whole runner | Defer |
| C4 | DFlash/DSpark drafter | Draft grows but target dominates | Memory unknown / feature audit / published weights | Public weights avoid train / yes | Target high, draft low | High; new masks/features/cache | Conditional independent audit | Do not select now |
| C5 | Larger/selective exact cache | Natural reuse yes, miss cost unresolved | Memory constrained / already old tested / yes | No / no | High | Medium | Yes but attribution blocked | Do not tune policy now |
| C6 | Context-padded whole graph | Strong fragmentation match | Conditional / collective agreement / mask audit | No / no | High | High; changed attention contract | Safety audit first | C1 has smaller contract |
| C7 | FlashInfer attention/plan | Small current attention share | Backend-dependent / local heads / possible | No / no | High | Medium-high numerics/layout | Yes | Insufficient local payoff |
| C8 | MLP backend/fusion | Strong compute share, counters missing | Kernel-dependent / shards / numerics | No BF16 / no | High | High kernel effort | One operator yes | After dispatch attribution |
| C9 | MPK/Blink | Strong conceptual launch match | Unqualified or missing SmartNIC / major rewrite | No / no if mapped | Low | Very high | Not matched whole runtime | Hardware/scope rejection |
| C10 | TP overlap / EP transport | Collectives frequent, wire cost unknown | SYS constraint / alters TP / EP wrong model | No TP / MoE change for EP | Medium | High deadlock/numerics | Poor immediate boundary | Attribution first |
| C11 | LMCache/Mooncake KV tier | No measured capacity/reuse pressure | Local tier possible / shard ownership / yes | No / no | Medium | High lifecycle | Local tier yes | Wrong current bottleneck |
| C12 | P/D distributed workers | No same-hardware serving case | Insufficient independent14B replicas | No / no | Medium | Very high | Not fixed deployment | Extra hardware needed |
| C13 | GPU acceptance/sampling | <=~1% combined current wall | Possible / status redesign / greedy possible | No / no | High | Medium-high semantics | Kernel yes, wall case weak | Retain safety barriers |
| C14 | Quantization/offload | Fits BF16 now; capacity not issue | Format/fabric-specific / kernel audit | Calibration possibly / often yes | Medium | High new contract | Different study | Out of present objective |

## Why exactly these two

Primary matches the strongest measured opportunity and keeps request semantics,
weights, attention and draft architecture. It is less hardware-specific than
MPK/FA4, less state-invasive than async serving or a parallel drafter, and less
context-sensitive than padding a whole verification graph. It is still allowed
to fail on copies, many region launches, capture memory and startup cost.

Secondary is a smaller runtime-stage alternative grounded in current MRV2,
not a second performance claim. Its evidence gap is explicit: metadata could
be too cheap at c<=4. If neither passes the gates, report that result rather
than forcing one of the remaining technologies into the project.

The best new discovery outside our historical agenda is DeepSpec's published
Qwen3-14B parallel drafts. It removes one availability objection, but does not
outweigh target-dominant profiling, new state/feature contracts and rank0 memory.
Compiler-free segmented replay is the more suitable *minimal transfer* found
in this scan; it differs from the previously tested dynamic exact-key cache.
