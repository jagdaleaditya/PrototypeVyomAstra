import json
import os
from datetime import datetime


class EventLogger:

    def __init__(self, experiment_name):

        self.experiment_name = experiment_name

        # Project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        # Log directory
        self.log_directory = os.path.join(
            project_root,
            "data",
            "experiment_logs"
        )

        os.makedirs(self.log_directory, exist_ok=True)

        # Create unique log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_file = os.path.join(
            self.log_directory,
            f"experiment_{timestamp}.json"
        )

        # Experiment data
        self.data = {
            "experiment": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "RUNNING",
            "events": []
        }

        self.save()

        print()
        print("==========================================")
        print("EVENT LOGGER STARTED")
        print("==========================================")
        print(f">>> EXPERIMENT: {self.experiment_name}")
        print(f">>> LOG FILE: {self.log_file}")
        print("==========================================")
        print()

    def log_event(self, activity, status="OK", message=""):

        event = {
            "timestamp": datetime.now().isoformat(),
            "activity": activity,
            "status": status,
            "message": message
        }

        self.data["events"].append(event)

        self.save()

        print(
            f">>> LOGGED: "
            f"{activity} | {status}"
        )

    def complete(self):

        self.data["end_time"] = datetime.now().isoformat()
        self.data["status"] = "COMPLETED"

        self.save()

        print()
        print(">>> EVENT LOG: EXPERIMENT COMPLETED")

    def warning(self, activity, expected):

        event = {
            "timestamp": datetime.now().isoformat(),
            "activity": activity,
            "status": "WARNING",
            "expected": expected
        }

        self.data["events"].append(event)

        self.save()

        print(
            f">>> LOGGED WARNING: "
            f"{activity} | Expected: {expected}"
        )

    def save(self):

        with open(
            self.log_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.data,
                file,
                indent=4
            )


if __name__ == "__main__":

    logger = EventLogger("Bottle Transfer")

    logger.log_event(
        "APPROACHING",
        "OK",
        "Astronaut approaching bottle"
    )

    logger.log_event(
        "PICKED",
        "OK",
        "Bottle picked"
    )

    logger.log_event(
        "MOVING",
        "OK",
        "Bottle moving"
    )

    logger.log_event(
        "RELEASED",
        "OK",
        "Bottle released"
    )

    logger.complete()

    print()
    print("Test log created successfully.")