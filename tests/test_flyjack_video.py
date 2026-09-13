import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly.flyjack_video import _latest_decision, _span
from chessfly.social_video import VideoRun


class FlyjackTimelineTests(unittest.TestCase):
    def test_span_finds_the_shot_containing_a_frame(self):
        spans = [
            {"name": "overview", "start_frame": 0, "end_frame": 10},
            {"name": "think", "start_frame": 10, "end_frame": 40},
        ]
        self.assertEqual(_span(spans, 0)["name"], "overview")
        self.assertEqual(_span(spans, 10)["name"], "think")
        self.assertEqual(_span(spans, 99)["name"], "think")

    def test_the_brain_holds_the_latest_neural_decision(self):
        moves = tuple(
            {"ply": ply, "actor": "chessfly" if ply % 2 else "stockfish"}
            for ply in range(1, 7)
        )
        run = VideoRun(None, {}, moves, (), (), ())
        self.assertEqual(_latest_decision(run, 0), 1)
        self.assertEqual(_latest_decision(run, 3), 3)
        self.assertEqual(_latest_decision(run, 5), 5)

    def test_no_decision_before_the_brain_has_moved(self):
        moves = ({"ply": 1, "actor": "stockfish"},)
        run = VideoRun(None, {}, moves, (), (), ())
        self.assertIsNone(_latest_decision(run, 0))


if __name__ == "__main__":
    unittest.main()
