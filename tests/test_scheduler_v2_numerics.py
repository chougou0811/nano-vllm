import json
from pathlib import Path
import tempfile
import unittest

import torch
from benchmarks.serving.scheduler_v2_correctness import POLICIES, RULES, PROBABILITY_RULES, compare_phase, scores
from benchmarks.serving.report import write_json
from benchmarks.serving.scheduler_v2_experiment import candidate_gate


class NumericalTests(unittest.TestCase):
    def setup_case(self, path, kind):
        reference = torch.linspace(-4, 4, 32)
        write_json(path/"acceptance-rules.json", RULES)
        write_json(path/"probability-rules.json", PROBABILITY_RULES)
        torch.save({(0,0): reference}, path/"hf.pt")
        for policy in POLICIES:
            actual = reference.clone()
            if policy == "slo-v2":
                if kind == "bad":
                    actual = -actual
                if kind == "nan":
                    actual[0] = float("nan")
            torch.save({(0,0): actual}, path/f"{policy}.pt")

    def test_pass_and_large_semantic_error_rejected(self):
        for kind, expected in [("good", 0), ("bad", 1), ("nan", 1)]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.setup_case(path, kind)
                self.assertEqual(compare_phase(path), expected)
                result = json.loads((path/"acceptance.json").read_text())
                self.assertEqual(result["passed"], expected == 0)
                self.assertEqual(compare_phase(path, probability=True), expected)

    def test_offset_invariance_not_raw_max_threshold(self):
        ref = torch.linspace(-8, 8, 128)
        result = scores(ref+2, ref)
        self.assertGreater(result["max_abs"], .5)
        self.assertLess(result["tv"], 1e-6)
        self.assertLess(result["relative_l2"], 1e-6)
        self.assertLess(result["weighted_rms"], 1e-6)

    def test_original_repeat_must_be_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.setup_case(path, "good")
            tensor = torch.load(path/"original-repeat.pt", weights_only=True)
            tensor[0,0][0] += .001
            torch.save(tensor, path/"original-repeat.pt")
            self.assertEqual(compare_phase(path), 1)

    def test_candidate_gate_does_not_hide_candidate_or_hard_failures(self):
        report = dict(finite=True, coverage=True, original_exact_parity=True, high_precision_pass=True,
                      rules=PROBABILITY_RULES, rows=[])
        row = dict(accepted=False, policy="slo-v1", control_valid=True, top1=1, reference_top1=1,
                   tv=.04, weighted_rms=.08, limits=dict(tv=.03, weighted_rms=.07))
        report["rows"] = [row]
        self.assertTrue(candidate_gate(report))
        row["policy"] = "slo-v2"
        self.assertFalse(candidate_gate(report))
        row["policy"] = "slo-v1"
        row["tv"] = .11
        self.assertFalse(candidate_gate(report))
        row["tv"] = .04
        row["top1"] = 2
        self.assertFalse(candidate_gate(report))


if __name__ == "__main__":
    unittest.main()
