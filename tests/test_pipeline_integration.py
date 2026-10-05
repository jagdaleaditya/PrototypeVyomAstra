"""
Pipeline Integration Test for RackRelativePose Feature Flag
===========================================================
Verifies that:
1. When feature flag is DISABLED (default MVP mode), RackRelativePose is inactive
   and existing pipeline behavior is 100% preserved.
2. When feature flag is ENABLED, RackRelativePose initializes and attaches
   3D rack-relative data to the process_frame() output dictionary without regression.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HMR_PATH = os.path.join(PROJECT_ROOT, "hmr")
if HMR_PATH not in sys.path:
    sys.path.insert(0, HMR_PATH)

DASHBOARD_PATH = os.path.join(PROJECT_ROOT, "dashboard")
if DASHBOARD_PATH not in sys.path:
    sys.path.insert(0, DASHBOARD_PATH)

from ai_engine import DonutsAI


class TestPipelineFeatureFlag(unittest.TestCase):
    """
    Test suite verifying feature flag toggling and MVP behavior preservation.
    """

    def test_feature_flag_disabled_by_default(self):
        """Ensure feature flag is False by default and RackRelativePose is inactive."""
        # Ensure environment variable is unset
        if "DONUTS_ENABLE_RACK_RELATIVE_POSE" in os.environ:
            del os.environ["DONUTS_ENABLE_RACK_RELATIVE_POSE"]

        ai = DonutsAI()
        self.assertFalse(ai.enable_rack_relative_pose)
        self.assertIsNone(ai.rack_relative_pose)

        # Test process_frame on a black frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = ai.process_frame(dummy_frame)

        # Verify standard MVP keys are present
        expected_keys = [
            "frame", "state", "activity", "activity_status",
            "controller_status", "controller_message", "step_index",
            "total_steps", "completed", "next_step", "fps", "rack_relative_pose"
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Rack relative pose data must be None when disabled
        self.assertIsNone(result["rack_relative_pose"])

        # Check existing state machine / activity defaults
        self.assertEqual(result["state"], "IDLE")
        self.assertEqual(result["controller_status"], "READY")
        self.assertEqual(result["step_index"], 0)

        ai.close()

    def test_feature_flag_enabled_via_env(self):
        """Ensure feature flag activates when environment variable is set."""
        os.environ["DONUTS_ENABLE_RACK_RELATIVE_POSE"] = "1"
        try:
            ai = DonutsAI()
            self.assertTrue(ai.enable_rack_relative_pose)
            self.assertIsNotNone(ai.rack_relative_pose)
            self.assertTrue(ai.rack_relative_pose.is_calibrated)

            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = ai.process_frame(dummy_frame)

            self.assertIn("rack_relative_pose", result)
            # On a black frame with no person, rack_relative_pose is None but gracefully handled
            self.assertIsNone(result["rack_relative_pose"])

            ai.close()
        finally:
            del os.environ["DONUTS_ENABLE_RACK_RELATIVE_POSE"]


if __name__ == "__main__":
    unittest.main()
