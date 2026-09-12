import unittest
from types import SimpleNamespace
from unittest.mock import patch

from methods.tagfex.tagfex import TagFex


class TagFexBackboneConfigTests(unittest.TestCase):
    def _initialise(self, backbone_configs, init_classes=100, inc_classes=20):
        learner = object.__new__(TagFex)
        learner.data_manager = SimpleNamespace(
            dataset_name="cub200",
            init_num_cls=init_classes,
            inc_num_cls=inc_classes,
        )
        learner.device = "cpu"
        learner.configs = {}
        captured = {}

        def fake_network(backbone, network, device):
            captured["backbone"] = backbone
            captured["network"] = network
            captured["device"] = device
            return object()

        with patch("methods.tagfex.tagfex.TagFexNet", side_effect=fake_network):
            learner._init_network(backbone_configs, {})
        return captured["backbone"]

    def test_explicit_small_base_override_is_preserved(self):
        backbone = self._initialise(
            {"name": "resnet18", "params": {"small_base": True}}
        )

        self.assertTrue(backbone["params"]["small_base"])
        self.assertEqual(backbone["params"]["dataset_name"], "cub200")

    def test_historical_automatic_choice_remains_the_default(self):
        unequal = self._initialise({"name": "resnet18"})
        equal = self._initialise(
            {"name": "resnet18"}, init_classes=20, inc_classes=20
        )

        self.assertFalse(unequal["params"]["small_base"])
        self.assertTrue(equal["params"]["small_base"])


if __name__ == "__main__":
    unittest.main()
