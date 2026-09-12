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

