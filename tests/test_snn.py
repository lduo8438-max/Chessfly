import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly import LIFNetwork, LIFParameters, Synapse


class LIFNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = LIFParameters(dt_ms=1.0, refractory_ms=2.0)

    def test_external_increment_can_trigger_spike(self) -> None:
        network = LIFNetwork(1, parameters=self.parameters)
        self.assertEqual(network.step({0: 20.0}), (0,))

    def test_synapse_is_delivered_after_configured_delay(self) -> None:
        network = LIFNetwork(
            2, (Synapse(0, 1, 20.0, delay_steps=2),), self.parameters
        )
        self.assertEqual(network.step({0: 20.0}), (0,))
        self.assertEqual(network.step(), ())
        self.assertEqual(network.step(), (1,))

    def test_refractory_period_blocks_immediate_respike(self) -> None:
        network = LIFNetwork(1, parameters=self.parameters)
        self.assertEqual(network.step({0: 20.0}), (0,))
        self.assertEqual(network.step({0: 20.0}), ())
        self.assertEqual(network.step({0: 20.0}), ())
        self.assertEqual(network.step({0: 20.0}), (0,))

    def test_reset_clears_pending_events(self) -> None:
        network = LIFNetwork(2, (Synapse(0, 1, 20.0),), self.parameters)
        network.step({0: 20.0})
        network.reset()
        self.assertEqual(network.step(), ())
        self.assertEqual(network.step_index, 1)

    def test_rejects_out_of_range_synapse(self) -> None:
        with self.assertRaises(ValueError):
            LIFNetwork(2, (Synapse(0, 2, 1.0),), self.parameters)


if __name__ == "__main__":
    unittest.main()
