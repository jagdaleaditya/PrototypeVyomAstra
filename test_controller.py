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