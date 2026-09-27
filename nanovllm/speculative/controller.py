"""Request-local measured-progress controller; no scheduler or KV operations."""
from collections import deque
from dataclasses import dataclass
import math


@dataclass
class Estimate:
    outputs: float
    proposal_ns: float
    verification_ns: float
    samples: int = 1

    @property
    def rate(self):
        return self.outputs / (self.proposal_ns + self.verification_ns)


class AdaptiveK:
    candidates = (1, 2, 3, 4, 5, 6)

    def __init__(self, alpha=0.25, probe_interval=12, switch_margin=0.10):
        if not 0 < alpha <= 1 or probe_interval < 1 or not math.isfinite(switch_margin) or switch_margin < 0:
            raise ValueError("Invalid controller parameters")
        self.alpha, self.probe_interval, self.switch_margin = alpha, probe_interval, switch_margin
        self.tables, self.incumbents, self.updates, self.probes = {}, {}, {}, {}
        self.history = deque(maxlen=16)
        self.early_rejections = 0
        self.calls = 0
        self.last_decision = {}

    def select(self, context, remaining):
        bucket = context // 512
        table = self.tables.setdefault(bucket, {})
        incumbent = self.incumbents.setdefault(bucket, 3)
        if self.calls == 0:
            chosen, reason = 3, "initial-prefix"
        elif unseen := [k for k in self.candidates if k not in table]:
            chosen, reason = unseen[0], "calibration"
        elif self.updates.get(bucket, 0) % self.probe_interval == 0:
            index = self.probes.get(bucket, 0)
            chosen, reason = self.candidates[index % len(self.candidates)], "periodic-probe"
            self.probes[bucket] = index + 1
        else:
            best = max(self.candidates, key=lambda k: (table[k].rate, -k))
            if table[best].rate > table[incumbent].rate * (1 + self.switch_margin):
                incumbent = self.incumbents[bucket] = best
            chosen, reason = incumbent, "measured-rate"
        self.calls += 1
        proposed = sum(s[1] for s in self.history)
        self.last_decision = dict(requested_k=chosen, reason=reason, context_bucket=bucket,
            incumbent=incumbent, scores_tokens_s={k:e.rate*1e9 for k,e in table.items()},
            recent_acceptance=sum(s[0] for s in self.history)/proposed if proposed else None,
            recent_accepted_per_verification=sum(s[0] for s in self.history)/len(self.history) if self.history else None,
            recent_effective_outputs=sum(s[2] for s in self.history)/len(self.history) if self.history else None,
            consecutive_early_rejections=self.early_rejections,
            estimates={k:dict(proposal_ns=e.proposal_ns,verification_ns=e.verification_ns,
                              outputs=e.outputs,samples=e.samples) for k,e in table.items()})
        return chosen

    def observe(self, *, context, requested_k, proposed, accepted, outputs,
                proposal_ns, verification_ns, initial_prefix=False, terminal=False):
        if not (0 <= accepted <= proposed <= 6 and outputs >= 1):
            raise ValueError("Invalid progress observation")
        if not all(math.isfinite(t) and t >= 0 for t in (proposal_ns,verification_ns)) or proposal_ns+verification_ns <= 0:
            raise ValueError("Invalid latency observation")
        self.history.append((accepted,proposed,outputs))
        self.early_rejections = self.early_rejections+1 if proposed and accepted == 0 else 0
        # Setup and output-clipped/terminal samples remain in the trace but are
        # not estimates of steady-state, unconstrained K-step progress.
        if initial_prefix or terminal or proposed != requested_k or not proposed:
            return
        bucket=context//512
        table=self.tables.setdefault(bucket,{})
        if proposed not in table:
            table[proposed]=Estimate(outputs,proposal_ns,verification_ns)
        else:
            entry=table[proposed]
            for name,value in (("outputs",outputs),("proposal_ns",proposal_ns),("verification_ns",verification_ns)):
                setattr(entry,name,getattr(entry,name)+self.alpha*(value-getattr(entry,name)))
            entry.samples += 1
        self.updates[bucket]=self.updates.get(bucket,0)+1
