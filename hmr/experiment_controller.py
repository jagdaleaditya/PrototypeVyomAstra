"""
DONUTS - Experiment Controller (v2)
====================================
Loads experiment definition from config/experiment.json.
Validates activity sequence, detects:
  - CORRECT step
  - WRONG_ORDER (wrong activity, but a valid later step — implying skip)
  - STEP_SKIPPED (detected activity corresponds to a step after the expected one)
  - WRONG_STEP (completely out-of-sequence or unknown activity)
  - STEP_TIMEOUT (configurable per-step deadline)
  - EXPERIMENT_COMPLETE

Voice alerts are injected as an optional AlertManager (duck-typed).
This keeps the controller decoupled from the audio system.

Backward compatible with existing DonutsAI usage:
  - process_activity() returns same dict structure as before
  - reset() works as before
  - get_expected_step() works as before
"""

import json
import os
import time
from typing import Optional, Any


class ExperimentController:

    def __init__(
        self,
        config_path: Optional[str] = None,
        alert_manager: Optional[Any] = None,
        step_timeout_seconds: float = 60.0,
    ):
        """
        Args:
            config_path:           Path to experiment.json. Auto-detected if None.
            alert_manager:         Optional AlertManager instance for voice alerts.
                                   Must expose .speak(), .alert_wrong_step(),
                                   .alert_step_skipped(), .alert_experiment_complete().
            step_timeout_seconds:  Seconds before a STEP_TIMEOUT event is raised.
        """
        self.alert_manager = alert_manager
        self.step_timeout_seconds = step_timeout_seconds

        # ---- Find project root ----
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if config_path is None:
            config_path = os.path.join(project_root, "config", "experiment.json")

        # ---- Load experiment configuration ----
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[ExperimentController] ERROR loading config: {e}")
            self.config = {
                "experiment_name": "Unknown",
                "steps": [],
                "enable_rack_relative_pose": False,
            }

        self.experiment_name = self.config.get("experiment_name", "Unknown")
        self.steps = self.config.get("steps", [])

        # Extract ordered activity list
        self.STEP_ORDER = [step["activity"] for step in self.steps]

        # Activity → human-readable name
        self.STEP_NAMES = {step["activity"]: step["name"] for step in self.steps}

        # Activity → step id
        self.STEP_IDS = {step["activity"]: step["id"] for step in self.steps}

        # State
        self.step_index = 0
        self.completed = False
        self.skipped_steps: list = []
        self.completed_steps: list = []
        self.wrong_steps: list = []

        # Timing
        self.experiment_start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None

        self._print_config()

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _print_config(self):
        print()
        print("==========================================")
        print("EXPERIMENT CONFIGURATION LOADED")
        print("==========================================")
        print(f">>> EXPERIMENT: {self.experiment_name}")
        print(f">>> STEPS: {len(self.STEP_ORDER)}")
        for step in self.steps:
            print(f">>> STEP {step['id']}: {step['name']}")
        print("==========================================")
        print()

    def _voice(self, message: str):
        """Fire voice alert if an AlertManager is available."""
        if self.alert_manager is not None:
            try:
                self.alert_manager.speak(message)
            except Exception as e:
                print(f"[ExperimentController] Alert error: {e}")
        else:
            # Fallback: try legacy voice_alert module (backward compat)
            try:
                from voice_alert import speak
                speak(message)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def get_expected_step(self) -> str:
        """Return the activity key of the next expected step, or 'COMPLETE'."""
        if self.completed:
            return "COMPLETE"
        if self.step_index >= len(self.STEP_ORDER):
            return "COMPLETE"
        return self.STEP_ORDER[self.step_index]

    def get_expected_step_name(self) -> str:
        """Return the human-readable name of the next expected step."""
        expected = self.get_expected_step()
        return self.STEP_NAMES.get(expected, expected)

    def get_step_name(self, activity: str) -> str:
        """Return the human-readable name for any activity key."""
        return self.STEP_NAMES.get(activity, activity)

    def start_experiment(self):
        """Mark the experiment as started (sets timing reference)."""
        self.experiment_start_time = time.time()
        self.step_start_time = time.time()
        print(f"[ExperimentController] EXPERIMENT STARTED: {self.experiment_name}")

    def check_timeout(self) -> bool:
        """
        Returns True if the current step has exceeded step_timeout_seconds.
        Call this periodically from the main loop.
        """
        if self.completed or self.step_start_time is None:
            return False
        elapsed = time.time() - self.step_start_time
        if elapsed > self.step_timeout_seconds:
            expected_name = self.get_expected_step_name()
            print(f"[ExperimentController] STEP TIMEOUT: {expected_name}")
            self._voice(f"Step timeout. Please complete {expected_name}.")
            return True
        return False

    def process_activity(self, activity: str) -> dict:
        """
        Validate an observed activity against the experiment sequence.

        Returns a dict with keys:
            accepted (bool)
            status   (str):  "OK" | "COMPLETE" | "WRONG_ORDER" | "STEP_SKIPPED" |
                             "WRONG_STEP" | "UNKNOWN"
            message  (str)
            expected (str):  expected activity key (on failure)
            detected (str):  detected activity key (on failure)
            skipped  (list): list of skipped activity keys (if STEP_SKIPPED)
            next_step (str): next expected activity key (on success)
        """
        if self.experiment_start_time is None:
            self.start_experiment()

        if activity not in self.STEP_ORDER:
            return {
                "accepted": False,
                "status": "UNKNOWN",
                "message": f"Unknown activity: {activity}",
                "expected": self.get_expected_step(),
                "detected": activity,
            }

        if self.completed:
            return {
                "accepted": False,
                "status": "COMPLETE",
                "message": "Experiment already completed.",
                "expected": "COMPLETE",
                "detected": activity,
            }

        expected = self.get_expected_step()
        activity_index = self.STEP_ORDER.index(activity)

        # ------------------------------------------------------------------
        # CORRECT STEP
        # ------------------------------------------------------------------
        if activity == expected:
            self.completed_steps.append(activity)
            self.step_index += 1
            self.step_start_time = time.time()

            print(f"\n>>> CONTROLLER: CORRECT STEP — {activity}")

            # Experiment completed?
            if self.step_index >= len(self.STEP_ORDER):
                self.completed = True
                print()
                print("==========================================")
                print(">>> EXPERIMENT COMPLETE")
                print("==========================================")
                self._voice("Experiment completed successfully.")
                return {
                    "accepted": True,
                    "status": "COMPLETE",
                    "message": "Experiment completed successfully",
                    "next_step": "COMPLETE",
                }

            next_step = self.get_expected_step()
            print(f">>> NEXT: {self.STEP_NAMES.get(next_step, next_step)}")

            return {
                "accepted": True,
                "status": "OK",
                "message": f"{activity} accepted",
                "next_step": next_step,
            }

        # ------------------------------------------------------------------
        # STEP SKIPPED — detected a later step
        # ------------------------------------------------------------------
        expected_index = self.STEP_ORDER.index(expected)

        if activity_index > expected_index:
            # Steps between expected and detected were skipped
            skipped = self.STEP_ORDER[expected_index:activity_index]
            self.skipped_steps.extend(skipped)

            detected_name = self.STEP_NAMES.get(activity, activity)
            expected_name = self.STEP_NAMES.get(expected, expected)
            skipped_names = [self.STEP_NAMES.get(s, s) for s in skipped]

            warning = (
                f"STEP_SKIPPED: {skipped_names} skipped. "
                f"Detected: {detected_name}. Expected: {expected_name}."
            )
            print(f"\n!!! {warning}")

            self._voice(
                f"Warning. Step skipped. Expected {expected_name}."
            )

            self.wrong_steps.append({
                "detected": activity,
                "expected": expected,
                "skipped": skipped,
                "status": "STEP_SKIPPED",
            })

            return {
                "accepted": False,
                "status": "STEP_SKIPPED",
                "message": warning,
                "expected": expected,
                "detected": activity,
                "skipped": skipped,
            }

        # ------------------------------------------------------------------
        # WRONG ORDER — detected an earlier step
        # ------------------------------------------------------------------
        detected_name = self.STEP_NAMES.get(activity, activity)
        expected_name = self.STEP_NAMES.get(expected, expected)

        warning = (
            f"WRONG_ORDER: {detected_name} detected. "
            f"Expected: {expected_name}."
        )
        print(f"\n!!! {warning}")

        self._voice(
            f"Warning. Wrong step. Expected {expected_name}. "
            f"Detected {detected_name}."
        )

        self.wrong_steps.append({
            "detected": activity,
            "expected": expected,
            "status": "WRONG_ORDER",
        })

        return {
            "accepted": False,
            "status": "WARNING",
            "message": warning,
            "expected": expected,
            "detected": activity,
        }

    def reset(self):
        """Reset for a new experiment cycle."""
        self.step_index = 0
        self.completed = False
        self.skipped_steps = []
        self.completed_steps = []
        self.wrong_steps = []
        self.experiment_start_time = None
        self.step_start_time = None

        print()
        print(">>> CONTROLLER RESET")
        print(">>> READY FOR NEW EXPERIMENT")