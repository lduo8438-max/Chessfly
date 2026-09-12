import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pyarrow as pa
import pyarrow.feather as feather

from chessfly.graph import summarize_connection_file


class ConnectionGraphTests(unittest.TestCase):
    def test_summary_filters_both_edge_endpoints(self):
        table = pa.table(
            {
                "body_pre": [1, 1, 3, 2],
                "body_post": [2, 3, 1, 1],
                "weight": [4, 5, 6, 7],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.feather"
            feather.write_feather(table, path)
            summary = summarize_connection_file(path, (1, 2))
        self.assertEqual(summary.raw_edges, 4)
        self.assertEqual(summary.retained_neurons, 2)
        self.assertEqual(summary.retained_edges, 2)
        self.assertEqual(summary.retained_synaptic_contacts, 11)


if __name__ == "__main__":
    unittest.main()
