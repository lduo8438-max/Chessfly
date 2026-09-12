import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from PIL import Image

from chessfly.compiled_graph import compile_retained_csr
from chessfly.retina import (
    BLUE,
    LUMINANCE,
    build_retina_projection,
    sample_retina_stimulus,
)


class RetinaProjectionTests(unittest.TestCase):
    def test_modal_column_projection_and_channels(self):
        annotations = pa.table(
            {
                "bodyId": [10, 11, 12, 13, 14, 15],
                "type": ["R1-R6", "R1-R6", "L1", "L1", "R8p", "R8y"],
                "rootSide": ["L", "R", "L", "R", "L", "R"],
                "assignedOlHex1": [None, None, 1.0, 4.0, None, None],
                "assignedOlHex2": [None, None, 1.0, 4.0, None, None],
            }
        )
        edges = pa.table(
            {
                "body_pre": [10, 11, 14, 15],
                "body_post": [12, 13, 12, 13],
                "weight": [6, 7, 8, 9],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.feather"
            feather.write_feather(edges, weights)
            compiled = root / "compiled"
            compile_retained_csr(weights, annotations["bodyId"].to_pylist(), compiled)
            projection = root / "retina"
            summary = build_retina_projection(annotations, compiled, projection)
            channels = np.load(projection / "channel.npy")
            uv = np.load(projection / "uv.npy")
        self.assertEqual(summary.r1_r6_mapped, 2)
        self.assertEqual(summary.r8p_mapped, 1)
        self.assertEqual(summary.r8y_mapped, 1)
        self.assertEqual(channels.tolist(), [LUMINANCE, LUMINANCE, BLUE, 1])
        self.assertTrue(np.all((uv >= 0) & (uv <= 1)))

    def test_sampling_uses_linear_rgb_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "uv.npy", np.asarray([[0.0, 0.0], [1.0, 0.0]]))
            np.save(
                root / "channel.npy", np.asarray([LUMINANCE, BLUE], dtype=np.uint8)
            )
            image = Image.new("RGB", (2, 1))
            image.putdata([(255, 255, 255), (255, 0, 128)])
            values = sample_retina_stimulus(image, root)
        self.assertAlmostEqual(float(values[0]), 1.0, places=6)
        self.assertGreater(float(values[1]), 0.21)
        self.assertLess(float(values[1]), 0.22)


if __name__ == "__main__":
    unittest.main()
