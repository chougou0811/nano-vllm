import unittest
from types import SimpleNamespace

from benchmarks.serving.eagle3_phase51_runtime import eligible, Phase51Runner


class RegionBoundaryTests(unittest.TestCase):
    def test_exact_uniform_shapes_only(self):
        for count in (1, 2, 4):
            self.assertTrue(eligible([dict(proposals=[1, 2, 3])] * count))
        for count in (0, 3, 5):
            self.assertFalse(eligible([dict(proposals=[1, 2, 3])] * count))
        self.assertFalse(eligible([dict(proposals=[1]), dict(proposals=[1, 2, 3])]))

    def runner(self):
        runner = object.__new__(Phase51Runner)
        original = lambda value: value
        layer = SimpleNamespace(mlp=SimpleNamespace(forward=original))
        runner.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
        runner.layer_count = 1
        graph = lambda value: value + 1
        runner.regions = {(0, 4): graph}
        return runner, layer, original, graph

    def test_scope_restores_after_exception(self):
        runner, layer, original, graph = self.runner()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with runner.mlp_scope([dict(proposals=[1, 2, 3])]):
                self.assertIs(layer.mlp.forward, graph)
                raise RuntimeError("intentional")
        self.assertIs(layer.mlp.forward, original)

    def test_ineligible_stays_original(self):
        runner, layer, original, _ = self.runner()
        with runner.mlp_scope([dict(proposals=[1])]):
            self.assertIs(layer.mlp.forward, original)

    def test_missing_registry_does_not_partially_install(self):
        runner, layer, original, _ = self.runner()
        runner.model.model.layers.append(SimpleNamespace(mlp=SimpleNamespace(forward=original)))
        runner.layer_count = 2
        with self.assertRaises(KeyError):
            with runner.mlp_scope([dict(proposals=[1, 2, 3])]):
                pass
        self.assertIs(layer.mlp.forward, original)


if __name__ == "__main__":
    unittest.main()
