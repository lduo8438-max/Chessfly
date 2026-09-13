import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly.brain_cloud import GROUPS, group_of
from chessfly.fly_model import parse_appearance, parse_pose


POSE = """joints:
  joint_A1A2 : 0.21846014367186156
  joint_LFFemur : -67.57373506986399

  joint_RFTibia : 51.350331169288935
"""

CONFIG = """something_else: 1
appearance:
  wing:
    apply_to: ["LWing", "RWing"]
    texture: null
    material:
      rgba: [0.8, 0.8, 0.9, 0.3]
  thorax:
    apply_to: ["Thorax"]
    texture:
      builtin: flat
      rgb1: [0.59, 0.39, 0.12]
      rgb2: [0.59, 0.39, 0.12]
    material:
      rgba: [1, 1, 1, 1]
  tarsus:
    apply_to: [
      "LFTarsus1", "LFTarsus2",
      "RHTarsus5"
    ]
    texture:
      builtin: flat
      rgb1: [0.71, 0.51, 0.24]
    material:
      rgba: [1, 1, 1, 0.5]
vision:
  fovy_per_eye: 157
"""


class FlyModelParsingTests(unittest.TestCase):
    def test_pose_reads_every_joint_in_degrees(self):
        joints = parse_pose(POSE)
        self.assertEqual(len(joints), 3)
        self.assertAlmostEqual(joints["joint_LFFemur"], -67.57373506986399)

    def test_an_empty_pose_is_refused(self):
        with self.assertRaises(ValueError):
            parse_pose("joints:\n")

    def test_untextured_parts_use_the_material_colour(self):
        parts = parse_appearance(CONFIG)
        self.assertEqual(parts["LWing"]["rgba"], [0.8, 0.8, 0.9, 0.3])

    def test_textured_parts_take_the_texture_colour_and_material_alpha(self):
        parts = parse_appearance(CONFIG)
        self.assertEqual(parts["Thorax"]["rgba"], [0.59, 0.39, 0.12, 1.0])
        self.assertEqual(parts["RHTarsus5"]["rgba"], [0.71, 0.51, 0.24, 0.5])

    def test_multi_line_target_lists_are_read(self):
        parts = parse_appearance(CONFIG)
        for name in ("LFTarsus1", "LFTarsus2", "RHTarsus5"):
            self.assertEqual(parts[name]["group"], "tarsus")

    def test_the_block_stops_at_the_next_top_level_key(self):
        self.assertNotIn("fovy_per_eye", parse_appearance(CONFIG))

    def test_a_config_without_appearance_is_refused(self):
        with self.assertRaises(ValueError):
            parse_appearance("vision:\n  fovy_per_eye: 157\n")


class BrainGroupTests(unittest.TestCase):
    def test_groups_follow_the_flyjack_colour_classes(self):
        cases = {
            ("cb_intrinsic", "KCab"): "mushroom body",
            ("cb_intrinsic", "MBON01"): "mushroom body",
            ("descending_neuron", "DNa02"): "descending",
            ("ol_sensory", "R7"): "sensory",
            ("ol_intrinsic", "Mi1"): "optic",
            ("visual_projection", "LC4"): "optic",
            ("cb_intrinsic", "FB4A"): "central",
            ("vnc_intrinsic", "IN08"): "other",
            (None, None): "other",
        }
        for (superclass, cell_type), expected in cases.items():
            self.assertEqual(GROUPS[group_of(superclass, cell_type)], expected)


if __name__ == "__main__":
    unittest.main()
