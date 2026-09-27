"""Frozen calibration prior plus censored acceptance posteriors; no online probes."""
from copy import deepcopy
import math


class PriorAdaptiveK:
    uses_catchup_cost = True
    candidates = tuple(range(1, 7))
    epoch_tokens = 64
    prior_strength = 16

    def __init__(self, prior):
        if prior.get('schema') != 1 or not prior.get('buckets'):
            raise ValueError('Expected schema=1 calibration prior')
        self.prior = deepcopy(prior)
        for bucket, data in self.prior['buckets'].items():
            if not str(bucket).isdigit():
                raise ValueError('Invalid context bucket')
            for name, count in (('acceptance_p',6),('proposal_work_ns',6),('verification_ns',6),('catchup_ns',7)):
                values=data.get(name,[])
                if len(values)!=count or any(not math.isfinite(v) or v<=0 for v in values):
                    raise ValueError('Invalid prior '+name)
            if any(p>=1 for p in data['acceptance_p']):
                raise ValueError('Acceptance probabilities must be strictly inside (0,1)')
        self.tables = {}
        self.initial_remaining = None
        self.epoch = -1
        self.current = 3
        self.last_decision = {}

    def _table(self, bucket):
        if bucket not in self.tables:
            source=min(self.prior['buckets'],key=lambda b:(abs(int(b)-bucket),int(b)))
            p=self.prior['buckets'][source]
            strength=self.prior_strength
            self.tables[bucket]=dict(source_bucket=int(source),
                alpha=[strength*v for v in p['acceptance_p']],
                beta=[strength*(1-v) for v in p['acceptance_p']],
                costs={name:[[v,strength] for v in p[name]]
                       for name in ('proposal_work_ns','verification_ns','catchup_ns')})
        return self.tables[bucket]

    def select(self, context, remaining):
        if self.initial_remaining is None:
            self.initial_remaining=remaining
        progress=self.initial_remaining-remaining
        epoch=progress//self.epoch_tokens
        table=self._table(context//512)
        probabilities=[a/(a+b) for a,b in zip(table['alpha'],table['beta'])]
        costs=table['costs']
        forecasts={}
        for k in self.candidates:
            survival=1.0
            expected_outputs=1.0
            future_catchup=0.0
            for a in range(k+1):
                mass=survival*(1-probabilities[a]) if a<k else survival
                future=self._table((context+1+a)//512)['costs']['catchup_ns'][a][0]
                future_catchup += mass*future
                if a<k:
                    survival *= probabilities[a]
                    expected_outputs += survival
            work=costs['proposal_work_ns'][k-1][0]
            verify=costs['verification_ns'][k-1][0]
            forecasts[k]=dict(outputs=expected_outputs,proposal_work_ns=work,verification_ns=verify,
                next_catchup_ns=future_catchup,rate_tokens_s=expected_outputs*1e9/(work+verify+future_catchup))
        refresh=epoch!=self.epoch
        if refresh:
            self.current=max(self.candidates,key=lambda k:(forecasts[k]['rate_tokens_s'],-k))
            self.epoch=epoch
        self.last_decision=dict(reason='prior-posterior-epoch' if refresh else 'epoch-hold',
            requested_k=self.current,epoch=epoch,context_bucket=context//512,source_bucket=table['source_bucket'],
            acceptance_posterior=probabilities,acceptance_alpha=list(table['alpha']),acceptance_beta=list(table['beta']),
            forecasts=forecasts,cost_observations={key:[pair[1]-self.prior_strength for pair in values]
                                                  for key,values in costs.items()},
            calibration_steps=0,probe_steps=0)
        return self.current

    def observe(self, *, context, requested_k, proposed, accepted, outputs,
                proposal_ns, verification_ns, initial_prefix=False, terminal=False,
                conditioning_ns=0, conditioning_rows=0):
        if not 0<=accepted<=proposed<=6 or outputs<1:
            raise ValueError('Invalid acceptance observation')
        if any(not math.isfinite(v) or v<0 for v in (proposal_ns,verification_ns,conditioning_ns)) or conditioning_ns>proposal_ns:
            raise ValueError('Invalid cost decomposition')
        if not proposed or terminal:
            return
        if not initial_prefix and not 1<=conditioning_rows<=7:
            raise ValueError('Invalid incremental catch-up row count')
        table=self._table(context//512)
        for depth in range(accepted):
            table['alpha'][depth] += 1
        if accepted<proposed:
            table['beta'][accepted] += 1
        # Unverified suffix is censored, not a set of additional failures.
        if initial_prefix:
            return
        def update(name,index,value):
            pair=table['costs'][name][index]
            pair[0]=(pair[0]*pair[1]+value)/(pair[1]+1)
            pair[1] += 1
        update('proposal_work_ns',proposed-1,proposal_ns-conditioning_ns)
        update('verification_ns',proposed-1,verification_ns)
        update('catchup_ns',conditioning_rows-1,conditioning_ns)
