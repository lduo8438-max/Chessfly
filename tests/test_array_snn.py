import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pyarrow as pa

from chessfly.array_snn import ArrayLIFNetwork, ArrayLIFParameters, transmitter_signs


class ArrayLIFNetworkTests(unittest.TestCase):
    def test_fixed_delay_delivers_signed_contact_current(self):
        parameters = ArrayLIFParameters(
            dt_ms=1.0,
            tau_membrane_ms=1.0,
            tau_synaptic_ms=1000.0,
            threshold_mv=-50.0,
            refractory_ms=1.0,
            delay_ms=2.0,
            contact_scale=1.0,
        )
        network = ArrayLIFNetwork(
            2,
            np.array([0]),
            np.array([1]),
            np.array([20]),
            np.array([1.0, 1.0]),
            parameters,
        )
        drive = np.array([20.0, 0.0], dtype=np.float32)
        self.assertEqual(network.step(drive).tolist(), [0])
        self.assertEqual(network.step().tolist(), [])
        self.assertEqual(network.step().tolist(), [1])

    def test_inhibitory_source_prevents_postsynaptic_spike(self):
        parameters = ArrayLIFParameters(
            dt_ms=1.0,
            tau_membrane_ms=1.0,
            tau_synaptic_ms=1000.0,
            threshold_mv=-50.0,
            refractory_ms=1.0,
            delay_ms=1.0,
            contact_scale=1.0,
        )
        network = ArrayLIFNetwork(
            2,
            np.array([0]),
            np.array([1]),
            np.array([20]),
            np.array([-1.0, 1.0]),
            parameters,
        )
        network.step(np.array([20.0, 0.0], dtype=np.float32))
        self.assertEqual(network.step().tolist(), [])
        self.assertLess(network.voltage_mv[1], parameters.resting_mv)

    def _chain_network(self):
        parameters = ArrayLIFParameters(dt_ms=1.0, delay_ms=3.0, contact_scale=1.0)
        return ArrayLIFNetwork(
            3,
            np.array([0, 1]),
            np.array([1, 2]),
            np.array([40, 40]),
            np.array([1.0, 1.0, 1.0]),
            parameters,
        )

    def test_checkpointed_state_reproduces_the_continuing_trajectory(self):
        drive = np.array([14.0, 0.0, 0.0], dtype=np.float32)
        original = self._chain_network()
        for _ in range(20):
            original.step(drive)
        state = original.state_dict()
        expected = [original.step(drive).tolist() for _ in range(30)]

        restored = self._chain_network()
        restored.load_state_dict(state)
        self.assertEqual(restored.step_index, 20)
        replayed = [restored.step(drive).tolist() for _ in range(30)]
        self.assertEqual(replayed, expected)

    def test_a_reset_network_does_not_reproduce_the_trajectory(self):
        drive = np.array([14.0, 0.0, 0.0], dtype=np.float32)
        original = self._chain_network()
        for _ in range(20):
            original.step(drive)
        expected = [original.step(drive).tolist() for _ in range(30)]
        fresh = self._chain_network()
        self.assertNotEqual([fresh.step(drive).tolist() for _ in range(30)], expected)

    def test_mismatched_checkpoint_is_rejected(self):
        state = self._chain_network().state_dict()
        state["voltage_mv"] = np.zeros(4, dtype=np.float32)
        with self.assertRaises(ValueError):
            self._chain_network().load_state_dict(state)
        incomplete = self._chain_network().state_dict()
        del incomplete["queue"]
        with self.assertRaises(ValueError):
            self._chain_network().load_state_dict(incomplete)

    def test_consensus_transmitter_policy_is_explicit(self):
        table = pa.table(
            {
                "body": [10, 20, 30, 40],
                "consensus_nt": ["acetylcholine", "gaba", "dopamine", "unclear"],
            }
        )
        values = transmitter_signs(table, np.array([10, 20, 30, 40]))
        np.testing.assert_array_equal(values, [1.0, -1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

