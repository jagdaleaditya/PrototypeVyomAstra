import sys
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
    QListWidget,
)

from ai_engine import DonutsAI


class DonutsDashboard(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "DONUTS - Experiment Monitoring System"
        )

        self.resize(1200, 750)
        self.setMinimumSize(1000, 650)
        self.setMaximumSize(1600, 950)

        # ==============================================
        # AI ENGINE
        # ==============================================

        self.ai = DonutsAI()

        # ==============================================
        # CAMERA
        # ==============================================

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            print("ERROR: Could not open webcam.")

            self.camera = None

        else:

            print(">>> CAMERA CONNECTED")

        # ==============================================
        # UI
        # ==============================================

        self.build_ui()

        # ==============================================
        # CAMERA + AI TIMER
        # ==============================================

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_system
        )

        self.timer.start(30)

    # ==================================================
    # BUILD UI
    # ==================================================

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        main_layout = QVBoxLayout(
            central
        )

        main_layout.setContentsMargins(
            20, 20, 20, 20
        )

        main_layout.setSpacing(15)

        # ==================================================
        # HEADER
        # ==================================================

        header = QHBoxLayout()

        title = QLabel("DONUTS")

        title.setFont(
            QFont(
                "Arial",
                28,
                QFont.Bold
            )
        )

        subtitle = QLabel(
            "AI-Powered Microgravity Experiment Monitoring"
        )

        subtitle.setFont(
            QFont(
                "Arial",
                13
            )
        )

        system_status = QLabel(
            "● SYSTEM ONLINE"
        )

        system_status.setFont(
            QFont(
                "Arial",
                13,
                QFont.Bold
            )
        )

        header.addWidget(title)

        header.addSpacing(20)

        header.addWidget(subtitle)

        header.addStretch()

        header.addWidget(system_status)

        main_layout.addLayout(header)

        # ==================================================
        # MAIN CONTENT
        # ==================================================

        content = QHBoxLayout()

        content.setSpacing(15)

        # ==================================================
        # CAMERA
        # ==================================================

        camera_frame = QFrame()

        camera_frame.setFrameShape(
            QFrame.StyledPanel
        )

        camera_layout = QVBoxLayout(
            camera_frame
        )

        camera_title = QLabel(
            "LIVE AI CAMERA"
        )

        camera_title.setFont(
            QFont(
                "Arial",
                16,
                QFont.Bold
            )
        )

        self.camera_label = QLabel()

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setMinimumSize(
            700,
            430
        )

        self.camera_label.setText(
            "INITIALIZING CAMERA..."
        )

        self.camera_label.setFont(
            QFont(
                "Arial",
                20,
                QFont.Bold
            )
        )

        camera_layout.addWidget(
            camera_title
        )

        camera_layout.addWidget(
            self.camera_label
        )

        content.addWidget(
            camera_frame,
            2
        )

        # ==================================================
        # INFORMATION PANEL
        # ==================================================

        info_frame = QFrame()

        info_frame.setFrameShape(
            QFrame.StyledPanel
        )

        info_layout = QVBoxLayout(
            info_frame
        )

        experiment_title = QLabel(
            "EXPERIMENT"
        )

        experiment_title.setFont(
            QFont(
                "Arial",
                12,
                QFont.Bold
            )
        )

        experiment_name = QLabel(
            "Bottle Transfer"
        )

        experiment_name.setFont(
            QFont(
                "Arial",
                20,
                QFont.Bold
            )
        )

        activity_title = QLabel(
            "CURRENT ACTIVITY"
        )

        activity_title.setFont(
            QFont(
                "Arial",
                12,
                QFont.Bold
            )
        )

        self.activity = QLabel(
            "WAITING"
        )

        self.activity.setFont(
            QFont(
                "Arial",
                22,
                QFont.Bold
            )
        )

        step_title = QLabel(
            "CURRENT STEP"
        )

        step_title.setFont(
            QFont(
                "Arial",
                12,
                QFont.Bold
            )
        )

        self.step = QLabel(
            "0 / 4"
        )

        self.step.setFont(
            QFont(
                "Arial",
                22,
                QFont.Bold
            )
        )

        next_title = QLabel(
            "NEXT STEP"
        )

        next_title.setFont(
            QFont(
                "Arial",
                12,
                QFont.Bold
            )
        )

        self.next_step = QLabel(
            "APPROACH BOTTLE"
        )

        self.next_step.setFont(
            QFont(
                "Arial",
                16,
                QFont.Bold
            )
        )

        self.next_step.setWordWrap(
            True
        )

        # Controller status

        controller_title = QLabel(
            "CONTROLLER"
        )

        controller_title.setFont(
            QFont(
                "Arial",
                12,
                QFont.Bold
            )
        )

        self.controller = QLabel(
            "READY"
        )

        self.controller.setFont(
            QFont(
                "Arial",
                14,
                QFont.Bold
            )
        )

        info_layout.addWidget(
            experiment_title
        )

        info_layout.addWidget(
            experiment_name
        )

        info_layout.addSpacing(15)

        info_layout.addWidget(
            activity_title
        )

        info_layout.addWidget(
            self.activity
        )

        info_layout.addSpacing(15)

        info_layout.addWidget(
            step_title
        )

        info_layout.addWidget(
            self.step
        )

        info_layout.addSpacing(15)

        info_layout.addWidget(
            next_title
        )

        info_layout.addWidget(
            self.next_step
        )

        info_layout.addSpacing(15)

        info_layout.addWidget(
            controller_title
        )

        info_layout.addWidget(
            self.controller
        )

        info_layout.addStretch()

        content.addWidget(
            info_frame,
            1
        )

        main_layout.addLayout(
            content
        )

        # ==================================================
        # EXPERIMENT SEQUENCE
        # ==================================================

        sequence_frame = QFrame()

        sequence_frame.setFrameShape(
            QFrame.StyledPanel
        )

        sequence_layout = QVBoxLayout(
            sequence_frame
        )

        sequence_title = QLabel(
            "EXPERIMENT SEQUENCE"
        )

        sequence_title.setFont(
            QFont(
                "Arial",
                15,
                QFont.Bold
            )
        )

        sequence = QHBoxLayout()
        sequence.setSpacing(10)

        # Fixed source names. Labels are rebuilt every frame so
        # symbols can never accumulate.
        self.sequence_names = [
            "APPROACH",
            "PICK",
            "MOVE",
            "RELEASE"
        ]

        self.sequence_labels = []

        for step_name in self.sequence_names:

            label = QLabel(
                f"○ {step_name}"
            )

            label.setFont(
                QFont(
                    "Arial",
                    13,
                    QFont.Bold
                )
            )

            label.setAlignment(
                Qt.AlignCenter
            )

            label.setMinimumWidth(150)
            label.setMaximumWidth(220)

            self.sequence_labels.append(label)
            sequence.addWidget(label)

        sequence_layout.addWidget(
            sequence_title
        )

        sequence_layout.addLayout(
            sequence
        )

        main_layout.addWidget(
            sequence_frame
        )

        # ==================================================
        # EVENT LOG
        # ==================================================

        log_frame = QFrame()

        log_frame.setFrameShape(
            QFrame.StyledPanel
        )

        log_layout = QVBoxLayout(
            log_frame
        )

        log_title = QLabel(
            "EVENT LOG"
        )

        log_title.setFont(
            QFont(
                "Arial",
                15,
                QFont.Bold
            )
        )

        self.event_log = QListWidget()

        self.event_log.addItem(
            "System initialized"
        )

        self.event_log.addItem(
            "AI engine starting..."
        )

        self.event_log.addItem(
            "Experiment: Bottle Transfer"
        )

        self.event_log.addItem(
            "Waiting for astronaut activity..."
        )

        log_layout.addWidget(
            log_title
        )

        log_layout.addWidget(
            self.event_log
        )

        main_layout.addWidget(
            log_frame
        )

        # ==================================================
        # STYLE
        # ==================================================

        self.setStyleSheet("""
            QMainWindow {
                background: #F5F7FA;
            }

            QFrame {
                background: white;
                border: 1px solid #D9DEE7;
                border-radius: 8px;
            }

            QLabel {
                color: #172033;
            }

            QListWidget {
                background: white;
                border: none;
                font-size: 13px;
            }
        """)

    # ==================================================
    # UPDATE SYSTEM
    # ==================================================

    def update_system(self):

        if self.camera is None:
            return

        ret, frame = self.camera.read()

        if not ret:

            self.camera_label.setText(
                "CAMERA ERROR"
            )

            return

        # ==============================================
        # AI PROCESSING
        # ==============================================

        result = self.ai.process_frame(
            frame
        )

        processed_frame = result[
            "frame"
        ]

        # ==============================================
        # UPDATE CAMERA
        # ==============================================

        rgb = cv2.cvtColor(
            processed_frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = (
            rgb.shape
        )

        bytes_per_line = (
            channels * width
        )

        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            image
        )

        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(
            pixmap
        )

        # ==============================================
        # UPDATE ACTIVITY
        # ==============================================

        self.activity.setText(
            result["activity"]
        )

        # ==============================================
        # UPDATE STEP
        # ==============================================

        step_index = result["step_index"]
        total_steps = result["total_steps"]
        completed = result["completed"]

        current_step = total_steps if completed else step_index

        self.step.setText(
            f"{current_step} / {total_steps}"
        )

        # ==============================================
        # UPDATE NEXT STEP
        # ==============================================

        next_step = result[
            "next_step"
        ]

        names = {
            "APPROACHING":
                "APPROACH BOTTLE",

            "PICKED":
                "PICK BOTTLE",

            "MOVING":
                "MOVE BOTTLE",

            "RELEASED":
                "RELEASE BOTTLE",

            "COMPLETE":
                "EXPERIMENT COMPLETE"
        }

        self.next_step.setText(
            names.get(
                next_step,
                next_step
            )
        )

        # ==============================================
        # UPDATE CONTROLLER
        # ==============================================

        self.controller.setText(
            result["controller_status"]
        )

        # ==============================================
        # UPDATE SEQUENCE
        # ==============================================

        current_index = result["step_index"]
        completed = result["completed"]

        for index, label in enumerate(self.sequence_labels):

            step_name = self.sequence_names[index]

            if completed or index < current_index:
                symbol = "✓"
            elif index == current_index:
                symbol = "●"
            else:
                symbol = "○"

            # Always build from the original step name.
            # Never read label.text(), so nothing can accumulate.
            label.setText(f"{symbol} {step_name}")

            label.setMinimumWidth(150)
            label.setMaximumWidth(220)

    # ==================================================
    # CLEANUP
    # ==================================================

    def closeEvent(
        self,
        event
    ):

        if self.timer.isActive():

            self.timer.stop()

        if self.camera is not None:

            self.camera.release()

        self.ai.close()

        event.accept()


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = DonutsDashboard()

    window.show()

    sys.exit(
        app.exec()
    )