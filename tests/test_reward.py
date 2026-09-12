import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly.reward import stockfish_shaped_reward


class RewardTests(unittest.TestCase):
    def test_deadband_is_neutral(self):
        signal = stockfish_shaped_reward(5, 14)
        self.assertEqual(signal.value, 0.0)
        self.assertEqual(signal.valence, "neutral")

    def test_reward_is_clipped(self):
        self.assertEqual(stockfish_shaped_reward(0, 500).value, 1.0)
        self.assertEqual(stockfish_shaped_reward(0, -500).value, -1.0)

    def test_terminal_component_is_additive(self):
        signal = stockfish_shaped_reward(0, 40, terminal_result="win")
        self.assertEqual(signal.value, 1.2)
        self.assertEqual(signal.terminal_component, 1.0)


if __name__ == "__main__":
    unittest.main()
