from pathlib import Path
import subprocess
import unittest

from nanovllm.speculative.controller import AdaptiveK


class AdaptiveTests(unittest.TestCase):
    def observation(self,k,outputs=2,cost=100,**overrides):
        return dict(context=255,requested_k=k,proposed=k,accepted=outputs-1,
                    outputs=outputs,proposal_ns=cost/2,verification_ns=cost/2,**overrides)

    def test_rate_not_acceptance_objective(self):
        c=AdaptiveK()
        c.calls=1
        for k in c.candidates:
            c.observe(**self.observation(k,outputs=k+1,cost=20 if k==2 else 100*k))
        self.assertEqual(c.select(255,100),2)
        self.assertEqual(c.last_decision['reason'],'measured-rate')

    def test_setup_and_terminal_not_cost_training(self):
        c=AdaptiveK()
        c.observe(**self.observation(3,initial_prefix=True))
        c.observe(**self.observation(3,terminal=True))
        sample=self.observation(3); sample.update(proposed=1)
        c.observe(**sample)
        self.assertFalse(c.tables)
        self.assertEqual(len(c.history),3)

    def test_rejection_and_context_isolation(self):
        c=AdaptiveK()
        for _ in range(4): c.observe(**self.observation(1,outputs=1))
        self.assertEqual(c.early_rejections,4)
        c.select(1024,32)
        self.assertEqual(c.last_decision['context_bucket'],2)
        self.assertEqual(c.last_decision['scores_tokens_s'],{})
        c.observe(**self.observation(2,outputs=2))
        self.assertEqual(c.early_rejections,0)

    def test_deterministic_observation_replay(self):
        def simulate():
            c=AdaptiveK(); choices=[]
            for step in range(100):
                k=c.select(255,1000)
                choices.append((k,c.last_decision['reason']))
                c.observe(**self.observation(k,outputs=k+1,cost=20 if k==4 else 100*k,
                                            initial_prefix=step==0))
            return choices
        first=simulate()
        self.assertEqual(first,simulate())
        self.assertEqual(set(k for k,reason in first if reason=='measured-rate'),{4})
        self.assertEqual(set(k for k,_ in first),set(range(1,7)))

    def test_hysteresis(self):
        c=AdaptiveK(); c.calls=1
        for k in c.candidates: c.observe(**self.observation(k,outputs=2 if k>1 else 1,cost=100))
        c.tables[0][4].outputs=2.1
        self.assertEqual(c.select(255,100),3)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): AdaptiveK(alpha=0)
        with self.assertRaises(ValueError): AdaptiveK(switch_margin=float('nan'))
        c=AdaptiveK()
        with self.assertRaises(ValueError): c.observe(**self.observation(1,outputs=3))
        with self.assertRaises(ValueError): c.observe(**self.observation(1,cost=float('nan')))

    def test_frozen_phase2_state_and_scheduler(self):
        for name in ('nanovllm/speculative/draft_state.py','nanovllm/speculative/runtime.py',
                     'nanovllm/speculative/draft.py','nanovllm/speculative/acceptance.py',
                     'nanovllm/engine/scheduler.py','nanovllm/engine/policy_scheduler.py',
                     'nanovllm/engine/progress_scheduler.py','nanovllm/engine/block_manager.py',
                     'nanovllm/engine/model_runner.py'):
            self.assertEqual(Path(name).read_bytes(),subprocess.check_output(['git','show','0ce2c1e:'+name]),name)


if __name__=='__main__': unittest.main()
