import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly.stockfish import StockfishConfig, strength_options


class StockfishStrengthTests(unittest.TestCase):
    def test_elo_mode_limits_strength_by_elo(self):
        options = strength_options(StockfishConfig(elo=1800))
        self.assertTrue(options["UCI_LimitStrength"])
        self.assertEqual(options["UCI_Elo"], 1800)
        self.assertNotIn("Skill Level", options)

    def test_skill_mode_turns_elo_limiting_off(self):
        # Stockfish ignores Skill Level while UCI_LimitStrength is on, so the
        # two levers must never be sent together.
        options = strength_options(StockfishConfig(skill_level=3))
        self.assertFalse(options["UCI_LimitStrength"])
        self.assertEqual(options["Skill Level"], 3)
        self.assertNotIn("UCI_Elo", options)

    def test_elo_floor_is_enforced_in_elo_mode(self):
        with self.assertRaises(ValueError):
            StockfishConfig(elo=900)
        with self.assertRaises(ValueError):
            StockfishConfig(elo=4000)

    def test_skill_level_range_is_enforced(self):
        for level in (-1, 21):
            with self.assertRaises(ValueError):
                StockfishConfig(skill_level=level)
        for level in (0, 20):
            self.assertEqual(StockfishConfig(skill_level=level).skill_level, level)

    def test_skill_mode_ignores_an_unused_elo(self):
        config = StockfishConfig(elo=900, skill_level=0)
        self.assertNotIn("UCI_Elo", strength_options(config))

    def test_resource_limits_must_be_positive(self):
        for kwargs in ({"movetime_ms": 0}, {"threads": 0}, {"hash_mb": 0}):
            with self.assertRaises(ValueError):
                StockfishConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
