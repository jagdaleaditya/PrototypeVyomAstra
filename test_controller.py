import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HMR_DIR = os.path.join(PROJECT_ROOT, "hmr")
if HMR_DIR not in sys.path:
    sys.path.insert(0, HMR_DIR)

from hmr.experiment_controller import ExperimentController


controller = ExperimentController()


print("==========================================")
print("EXPERIMENT CONTROLLER TEST")
print("==========================================")


sequence = [
    "APPROACHING",
    "MOVING",
    "PICKED",
    "RELEASED"
]


for activity in sequence:

    print()
    print(f"Detected activity: {activity}")

    result = controller.process_activity(activity)

    print("Result:", result)


print()
print("==========================================")
print("TEST FINISHED")
print("==========================================")