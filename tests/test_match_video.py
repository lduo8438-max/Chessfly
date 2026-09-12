import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from chessfly.match_video import HOLD_PLIES, _grid, _pacing, _plate, _ply_index


class PacingTests(unittest.TestCase):
    def test_closing_plies_hold_longer_than_the_middlegame(self):
        weights = _pacing(20)
        spans = np.diff(np.concatenate(([0.0], weights)))
        self.assertAlmostEqual(float(spans.sum()), 1.0)
        for offset in range(1, HOLD_PLIES + 1):
            self.assertGreater(spans[-offset], spans[10] * 4)

    def test_every_ply_still_gets_screen_time(self):
        weights = _pacing(66)
        spans = np.diff(np.concatenate(([0.0], weights)))
        self.assertTrue(bool(np.all(spans > 0)))

    def test_a_single_ply_run_is_allowed(self):
        self.assertEqual(_ply_index(_pacing(1), 0.5), 0)

    def test_rejects_an_empty_run(self):
        with self.assertRaises(ValueError):
            _pacing(0)

    def test_fractions_map_forward_and_end_on_the_final_ply(self):
        weights = _pacing(12)
        self.assertEqual(_ply_index(weights, 0.0), 0)
        self.assertEqual(_ply_index(weights, 1.0), 11)
        seen = [_ply_index(weights, step / 400) for step in range(401)]
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(sorted(set(seen)), list(range(12)))


class PlateTests(unittest.TestCase):
    def test_plates_ping_pong_instead_of_cutting(self):
        plates = ["a", "b", "c", "d"]
        order = [_plate(plates, step / 30.0) for step in range(8)]
        self.assertEqual(order, ["a", "b", "c", "d", "c", "b", "a", "b"])

    def test_a_single_plate_is_held(self):
        self.assertEqual(_plate(["only"], 12.3), "only")


class GridTests(unittest.TestCase):
    def test_points_stay_inside_their_column(self):
        points = _grid(96, 100.0, 300.0, 40.0, 160.0)
        self.assertEqual(len(points), 96)
        self.assertTrue(all(100.0 <= x <= 300.0 for x, _ in points))
        self.assertTrue(all(40.0 <= y <= 160.0 for _, y in points))

    def test_an_empty_group_places_nothing(self):
        self.assertEqual(_grid(0, 0.0, 10.0, 0.0, 10.0), [])


if __name__ == "__main__":
    unittest.main()
