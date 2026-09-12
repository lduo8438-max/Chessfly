import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pyarrow as pa

from chessfly.annotations import (
    descending_body_ids,
    retained_neuron_body_ids,
    summarize_annotations,
    visual_candidate_body_ids,
)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.annotations = pa.table(
            {
                "bodyId": [10, 11, 12, 13],
                "type": ["R1-R6", "R8p", "DNp01", None],
                "instance": ["R1-R6_L", "R8p_R", "DNp01_L", None],
                "superclass": [
                    "ol_sensory",
                    "ol_sensory",
                    "descending_neuron",
                    None,
                ],
                "somaSide": ["L", "R", "L", None],
                "status": ["Traced", "Traced", "Traced", None],
            }
        )
        self.transmitters = pa.table(
            {
                "body": [10, 11, 12],
                "predicted_nt_confidence": [0.9, 0.8, 0.7],
                "predicted_nt": ["histamine", "histamine", "acetylcholine"],
                "consensus_nt": ["histamine", "histamine", "acetylcholine"],
            }
        )

    def test_summary_counts_roles(self):
        summary = summarize_annotations(self.annotations, self.transmitters)
        self.assertEqual(summary.neurons, 4)
        self.assertEqual(summary.descending_neurons, 1)
        self.assertEqual(summary.r1_r6_candidates, 1)
        self.assertEqual(summary.r8_candidates, 1)

    def test_body_id_selectors(self):
        self.assertEqual(descending_body_ids(self.annotations), (12,))
        self.assertEqual(visual_candidate_body_ids(self.annotations), (10, 11))
        self.assertEqual(retained_neuron_body_ids(self.annotations), (10, 11, 12))


if __name__ == "__main__":
    unittest.main()
