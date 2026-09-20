import unittest
import torch
from benchmarks.serving.eagle3_equivalence import Capture, dense_attention, stats, math_norm
from types import SimpleNamespace


class EquivalenceProbeTests(unittest.TestCase):
    def test_right_aligned_causal_reference(self):
        torch.manual_seed(14)
        q = torch.randn(4,4,8,dtype=torch.float64)
        k = torch.randn(9,2,8,dtype=torch.float64)
        v = torch.randn_like(k)
        parallel = dense_attention(q,k,v,8**-0.5)
        serial = torch.cat([dense_attention(q[i:i+1],k[:6+i],v[:6+i],8**-0.5) for i in range(4)])
        torch.testing.assert_close(parallel,serial,rtol=1e-12,atol=1e-12)
        poisoned = v.clone()
        poisoned[6:] = 1e6
        torch.testing.assert_close(dense_attention(q,k,poisoned,8**-0.5)[0],parallel[0],rtol=1e-12,atol=1e-12)

    def test_shared_rope_is_layer_gated(self):
        capture = Capture.__new__(Capture)
        capture.data = {}
        capture.active_layer = '02'
        args = (torch.arange(4),)
        output = (torch.ones(4,2,8),torch.ones(4,1,8))
        capture.rope('01',args,output)
        self.assertEqual(capture.data,{})
        capture.rope('02',args,output)
        self.assertEqual(capture.data['02.rope_q'][0].shape,(4,2,8))

    def test_stats_do_not_hide_nonfinite_or_mismatch(self):
        a = torch.ones(3)
        self.assertEqual(stats(a,a)['unequal'],0)
        self.assertEqual(stats(a,a+1)['max_abs'],1)
        self.assertFalse(stats(a,torch.full_like(a,float('nan')))['finite'])

    def test_fp32_norm_preserves_residual_and_input(self):
        x = torch.tensor([[1.,2.,3.,4.]])
        residual = torch.tensor([[2.,3.,4.,5.]])
        original_x,original_residual = x.clone(),residual.clone()
        module = SimpleNamespace(eps=1e-6,weight=torch.ones(4))
        norm,base = math_norm(module,x,residual)
        torch.testing.assert_close(base,original_x+original_residual)
        torch.testing.assert_close(x,original_x)
        torch.testing.assert_close(residual,original_residual)
        torch.testing.assert_close(norm,base*torch.rsqrt(base.square().mean(-1,keepdim=True)+module.eps))


if __name__ == '__main__':
    unittest.main()
