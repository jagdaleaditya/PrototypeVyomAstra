# ============================================================
# DONUTS - Activity Recognition
# ============================================================

class ActivityRecognizer:
    """
    Lightweight prototype activity recognizer.

    Compatible with:
      set_state(state)
      event(state)
      update(state)

    A completed cycle automatically resets when the system returns
    to IDLE, allowing the next bottle interaction to start normally.
    """

    STEP_ORDER = [
        "APPROACHING",
        "PICKED",
        "MOVING",
        "RELEASED",
    ]

    ACTIVITY_NAMES = {
        "IDLE": "WAITING",
        "APPROACHING": "APPROACHING BOTTLE",
        "PICKED": "HOLDING BOTTLE",
        "MOVING": "MOVING BOTTLE",
        "RELEASED": "BOTTLE RELEASED",
    }

    def __init__(self):
        self.last_state = "IDLE"
        self.step_index = 0
        self.completed = False
        self.message = "WAITING"
        self.status = "WAITING"

    def set_state(self, state):
        # Ignore repeated frames of the same state.
        if state == self.last_state:
            return self.ACTIVITY_NAMES.get(state, state)

        # After a complete cycle, IDLE means the bottle was released
        # and the system is ready to begin a NEW cycle.
        if state == "IDLE":
            if self.completed:
                self.step_index = 0
                self.completed = False
                self.message = "WAITING"
                self.status = "READY FOR NEXT EXPERIMENT"
                print(">>> ACTIVITY: READY FOR NEXT EXPERIMENT")

            self.last_state = "IDLE"
            return self.message

        self.last_state = state

        if state not in self.STEP_ORDER:
            self.message = self.ACTIVITY_NAMES.get(state, state)
            return self.message

        # Safety guard: never index beyond the final step.
        if self.step_index >= len(self.STEP_ORDER):
            self.step_index = 0
            self.completed = False

        expected = self.STEP_ORDER[self.step_index]

        if state == expected:
            self.step_index += 1
            self.message = self.ACTIVITY_NAMES[state]
            self.status = f"STEP {self.step_index}/4 OK"

            print(f">>> ACTIVITY: {self.message}")
            print(f">>> {self.status}")

            if self.step_index >= len(self.STEP_ORDER):
                self.completed = True
                self.status = "EXPERIMENT CYCLE COMPLETE"
                print(">>> EXPERIMENT CYCLE COMPLETE")

        else:
            self.message = f"UNEXPECTED: {self.ACTIVITY_NAMES[state]}"
            self.status = f"EXPECTED: {expected}"
            print(f">>> WARNING: {self.message}")
            print(f">>> {self.status}")

        return self.message

    def event(self, state):
        self.set_state(state)
        return self.status

    def update(self, state):
        return self.set_state(state)

    def reset_cycle(self):
        self.last_state = "IDLE"
        self.step_index = 0
        self.completed = False
        self.message = "WAITING"
        self.status = "WAITING"
