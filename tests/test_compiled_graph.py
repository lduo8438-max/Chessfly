import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from chessfly.compiled_graph import build_path_subgraph, compile_retained_csr


class CompiledGraphTests(unittest.TestCase):
    def test_compile_and_extract_path_subgraph(self):
        table = pa.table(
            {
                "body_pre": [10, 11, 12, 10, 13],
                "body_post": [11, 12, 14, 13, 14],
                "weight": [5, 6, 7, 2, 8],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.feather"
            feather.write_feather(table, weights)
            compiled = root / "compiled"
            summary = compile_retained_csr(weights, (10, 11, 12, 13, 14), compiled)
            self.assertEqual(summary.edges, 5)
            subgraph = root / "subgraph"
            path_summary = build_path_subgraph(
                compiled, (10,), (14,), subgraph, maximum_hops=3, minimum_weight=5
            )
            self.assertEqual(path_summary.neurons, 4)
            self.assertEqual(path_summary.edges, 3)
            self.assertEqual(
                np.load(subgraph / "node_ids.npy").tolist(), [10, 11, 12, 14]
            )


if __name__ == "__main__":
    unittest.main()
