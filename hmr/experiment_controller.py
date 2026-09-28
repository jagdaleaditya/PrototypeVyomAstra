import json
import os

from voice_alert import speak


class ExperimentController:

    def __init__(self):

        # Find project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Configuration file
        config_path = os.path.join(
            project_root,
            "config",
            "experiment.json"
        )

        # Load experiment configuration
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.experiment_name = self.config["experiment_name"]
        self.steps = self.config["steps"]

        # Extract activity order
        self.STEP_ORDER = [
            step["activity"]
            for step in self.steps
        ]

        # Activity → human-readable name
        self.STEP_NAMES = {
            step["activity"]: step["name"]
            for step in self.steps
        }

        self.step_index = 0
        self.completed = False

        print()
        print("==========================================")
        print("EXPERIMENT CONFIGURATION LOADED")
        print("==========================================")
        print(f">>> EXPERIMENT: {self.experiment_name}")
        print(f">>> STEPS: {len(self.STEP_ORDER)}")

        for step in self.steps:
            print(
                f">>> STEP {step['id']}: "
                f"{step['name']}"
            )

        print("==========================================")
        print()

    def get_expected_step(self):

        if self.completed:
            return "COMPLETE"

        return self.STEP_ORDER[self.step_index]

    def process_activity(self, activity):

        if activity not in self.STEP_ORDER:

            return {
                "accepted": False,
                "status": "UNKNOWN",
                "message": f"Unknown activity: {activity}"
            }

        expected = self.get_expected_step()

        # ------------------------------------------
        # CORRECT STEP
        # ------------------------------------------

        if activity == expected:

            self.step_index += 1

            print()
            print(">>> CONTROLLER: CORRECT STEP")
            print(f">>> DETECTED: {activity}")

            # Experiment completed
            if self.step_index >= len(self.STEP_ORDER):

                self.completed = True

                print()
                print("==========================================")
                print(">>> EXPERIMENT COMPLETE")
                print("==========================================")

                return {
                    "accepted": True,
                    "status": "COMPLETE",
                    "message": "Experiment completed successfully"
                }

            # Next expected step
            next_step = self.get_expected_step()

            print(
                f">>> NEXT: "
                f"{self.STEP_NAMES[next_step]}"
            )

            return {
                "accepted": True,
                "status": "OK",
                "message": f"{activity} accepted",
                "next_step": next_step
            }

        # ------------------------------------------
        # WRONG STEP
        # ------------------------------------------

        expected_name = self.STEP_NAMES.get(
            expected,
            expected
        )

        detected_name = self.STEP_NAMES.get(
            activity,
            activity
        )

        warning = (
            f"WARNING: {detected_name} detected. "
            f"Expected: {expected_name}"
        )

        print()
        print("!!! EXPERIMENT WARNING !!!")
        print(warning)

        # Voice alert
        speak(
            f"Warning. Wrong step. "
            f"Expected {expected_name}. "
            f"Detected {detected_name}."
        )

        return {
            "accepted": False,
            "status": "WARNING",
            "message": warning,
            "expected": expected,
            "detected": activity
        }

    def reset(self):

        self.step_index = 0
        self.completed = False

        print()
        print(">>> CONTROLLER RESET")
        print(">>> READY FOR NEW EXPERIMENT")