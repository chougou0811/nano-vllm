from copy import deepcopy
import unittest

from nanovllm.speculative.prior_controller import PriorAdaptiveK


def prior():
    return dict(schema=1,buckets={'0':dict(acceptance_p=[.7]*6,
        proposal_work_ns=[10,20,30,40,50,60],verification_ns=[100]*6,
        catchup_ns=[5,10,15,20,25,30,35])})


def observation(**kwargs):
    values=dict(context=255,requested_k=6,proposed=6,accepted=2,outputs=3,
                proposal_ns=100,verification_ns=100,conditioning_ns=60,conditioning_rows=3)
    values.update(kwargs)
    return values


class PriorTests(unittest.TestCase):
    def test_expected_progress_and_future_cost(self):
        c=PriorAdaptiveK(prior()); c.select(255,63)
        for k,f in c.last_decision['forecasts'].items():
            expected=1+sum(.7**j for j in range(1,k+1))
            self.assertAlmostEqual(f['outputs'],expected,places=12)
            self.assertAlmostEqual(f['next_catchup_ns'],5*expected,places=12)

    def test_censored_suffix(self):
        c=PriorAdaptiveK(prior()); t=c._table(0); a=t['alpha'][:]; b=t['beta'][:]
        c.observe(**observation())
        self.assertEqual(t['alpha'],[a[0]+1,a[1]+1]+a[2:])
        self.assertEqual(t['beta'],b[:2]+[b[2]+1]+b[3:])

    def test_full_accept_and_rejection(self):
        c=PriorAdaptiveK(prior()); t=c._table(0); b=t['beta'][:]
        c.observe(**observation(accepted=6,outputs=7))
        self.assertEqual(t['beta'],b)
        c.observe(**observation(accepted=0,outputs=1))
        self.assertEqual(t['beta'],[b[0]+1]+b[1:])

    def test_short_horizon_holds_without_probe(self):
        c=PriorAdaptiveK(prior()); first=c.select(255,63)
        for remaining in range(62,0,-1):
            c.observe(**observation(accepted=0,outputs=1))
            self.assertEqual(c.select(255+63-remaining,remaining),first)
            self.assertEqual(c.last_decision['reason'],'epoch-hold')
            self.assertEqual(c.last_decision['probe_steps'],0)

    def test_epoch_limit_and_replay(self):
        def replay():
            c=PriorAdaptiveK(prior()); result=[]; refresh=0
            for remaining in range(511,0,-1):
                k=c.select(1024+511-remaining,remaining)
                result.append(k)
                refresh += c.last_decision['reason']=='prior-posterior-epoch'
                c.observe(**observation(context=1024+511-remaining,accepted=remaining%7,outputs=1+remaining%7))
            self.assertEqual(refresh,8)
            return result
        self.assertEqual(replay(),replay())

    def test_credit_assignment(self):
        a=PriorAdaptiveK(prior()); b=PriorAdaptiveK(prior())
        a.observe(**observation(proposal_ns=100,conditioning_ns=60,conditioning_rows=7))
        b.observe(**observation(proposal_ns=50,conditioning_ns=10,conditioning_rows=1))
        self.assertEqual(a._table(0)['costs']['proposal_work_ns'],b._table(0)['costs']['proposal_work_ns'])
        self.assertNotEqual(a._table(0)['costs']['catchup_ns'],b._table(0)['costs']['catchup_ns'])

    def test_setup_cost_exclusion_but_acceptance_learning(self):
        c=PriorAdaptiveK(prior()); t=c._table(0); costs=deepcopy(t['costs'])
        c.observe(**observation(initial_prefix=True,conditioning_rows=1024))
        self.assertEqual(t['costs'],costs)
        self.assertNotEqual(t['alpha'],[.7*16]*6)

    def test_prior_immutable_and_context_fallback(self):
        p=prior(); before=deepcopy(p); c=PriorAdaptiveK(p)
        c.observe(**observation()); c.select(1536,128)
        self.assertEqual(c.last_decision['source_bucket'],0)
        self.assertEqual(p,before)
        self.assertEqual(PriorAdaptiveK(p)._table(0)['alpha'],[.7*16]*6)

    def test_invalid_costs(self):
        p=prior(); p['buckets']['0']['acceptance_p'][0]=1
        with self.assertRaises(ValueError): PriorAdaptiveK(p)
        c=PriorAdaptiveK(prior())
        with self.assertRaises(ValueError): c.observe(**observation(conditioning_ns=101))
        with self.assertRaises(ValueError): c.observe(**observation(verification_ns=float('nan')))
        with self.assertRaises(ValueError): c.observe(**observation(conditioning_rows=0))


if __name__=='__main__': unittest.main()
