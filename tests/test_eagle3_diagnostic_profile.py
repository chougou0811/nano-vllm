"""Phase 4.3 analysis tests; frozen correctness tests are untouched."""
import unittest

from benchmarks.serving.eagle3_phase43_summary import (
    analyze_trace, duration, gpu_category, intersection, merge_intervals,
)


class DiagnosticAnalysisTests(unittest.TestCase):
    def test_interval_union_not_kernel_sum(self):
        self.assertEqual(merge_intervals([(0, 4), (2, 6), (8, 9)]), [[0, 6], [8, 9]])
        self.assertEqual(duration([(0, 4), (2, 6)]), 6)

    def test_overlap_is_intersection_of_unions(self):
        self.assertEqual(intersection([(0, 5), (3, 7)], [(2, 4), (6, 8)]), 3)
        self.assertEqual(intersection([], [(0, 4)]), 0)

    def test_nccl_not_charged_to_enclosing_projection(self):
        kernel = dict(name="ncclDevKernel_AllReduce", cat="kernel")
        scope = dict(name="diag.module::mlp_down::layer.0.down_proj")
        self.assertEqual(gpu_category(kernel, scope), "communication")

    def test_external_id_attribution_and_idle(self):
        def event(name, cat, ts, dur, **args):
            return dict(ph="X", name=name, cat=cat, ts=ts, dur=dur, tid=1, args=args)
        events = [event("diag.target.0", "user_annotation", 0, 100),
                  event("diag.module::qkv::layer.qkv_proj", "user_annotation", 2, 20),
                  event("aten::mm", "cpu_op", 3, 2, **{"External id": 1}),
                  event("c10d::allreduce_", "cpu_op", 25, 2, **{"External id": 2}),
                  event("gemm", "kernel", 10, 10, **{"External id": 1}),
                  event("ncclDevKernel", "kernel", 30, 10, **{"External id": 2}),
                  event("draft_kernel", "kernel", 200, 10, **{"External id": 99})]
        result = analyze_trace(dict(traceEvents=events))["targets"][0]
        self.assertEqual(result["gpu_envelope_us"], 30)
        self.assertEqual(result["gpu_idle_gap_us"], 10)
        self.assertEqual(result["kernel_count"], 2)
        self.assertEqual(result["categories"]["qkv"]["sum_us"], 10)
        self.assertEqual(result["communication_union_us"], 10)

    def test_missing_gpu_correlation_fails_closed(self):
        with self.assertRaises(ValueError):
            analyze_trace(dict(traceEvents=[dict(ph="X", cat="user_annotation", tid=1,
                name="diag.target.0", ts=0, dur=100)]))


if __name__ == "__main__":
    unittest.main()
