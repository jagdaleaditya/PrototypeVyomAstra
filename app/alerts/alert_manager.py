"""
DONUTS - Alert Manager
======================
Manages voice alerts with:
  - Offline TTS via pyttsx3
  - Asynchronous (non-blocking) speech in a background thread
  - Per-alert cooldown to prevent spam
  - Enable/disable toggle
  - Alert priority categories
"""

import time
import threading
from typing import Optional, Dict
from enum import Enum


class AlertPriority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertManager:
    """
    Non-blocking voice alert system with cooldown management.

    Usage:
        alerts = AlertManager(voice_enabled=True, cooldown_seconds=5)
        alerts.speak("Warning. Step skipped.")
        alerts.speak("Step skipped.")   # ignored within cooldown window
    """

    def __init__(
        self,
        voice_enabled: bool = True,
        cooldown_seconds: float = 5.0,
        voice_rate: int = 160,
        voice_volume: float = 1.0,
    ):
        self.voice_enabled = voice_enabled
        self.cooldown_seconds = cooldown_seconds
        self._last_spoken: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._speech_thread: Optional[threading.Thread] = None
        self._engine = None
        self._engine_lock = threading.Lock()

        if voice_enabled:
            self._init_engine(voice_rate, voice_volume)

    # ------------------------------------------------------------------
    # ENGINE INIT
    # ------------------------------------------------------------------

    def _init_engine(self, rate: int, volume: float):
        """Initialize pyttsx3 in a safe way — engine must be created on the
        thread where it will be used. We defer creation to the speech thread."""
        self._voice_rate = rate
        self._voice_volume = volume
        # Engine will be created lazily on the speech thread
        print("[AlertManager] Voice alert system ready (async pyttsx3).")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def speak(
        self,
        message: str,
        priority: AlertPriority = AlertPriority.NORMAL,
        force: bool = False,
    ) -> bool:
        """
        Speak a message asynchronously.

        Args:
            message:  Text to speak.
            priority: Alert priority (HIGH/CRITICAL bypass the shorter cooldown).
            force:    If True, bypass cooldown check entirely.

        Returns:
            True if the message was queued, False if suppressed by cooldown.
        """
        if not self.voice_enabled:
            return False

        # Determine effective cooldown
        effective_cooldown = self.cooldown_seconds
        if priority in (AlertPriority.HIGH, AlertPriority.CRITICAL):
            effective_cooldown = max(1.0, self.cooldown_seconds / 2)

        # Cooldown check
        now = time.time()
        with self._lock:
            last_time = self._last_spoken.get(message, 0.0)
            if not force and (now - last_time) < effective_cooldown:
                return False
            self._last_spoken[message] = now

        # Log
        print(f"[AlertManager] VOICE [{priority.value}]: {message}")

        # Launch or wait for background thread
        self._launch_speech(message)
        return True

    def speak_warning(self, message: str) -> bool:
        """Convenience: speak with NORMAL priority."""
        return self.speak(message, AlertPriority.NORMAL)

    def speak_error(self, message: str) -> bool:
        """Convenience: speak with HIGH priority."""
        return self.speak(message, AlertPriority.HIGH)

    def speak_critical(self, message: str) -> bool:
        """Convenience: speak with CRITICAL priority, force = True."""
        return self.speak(message, AlertPriority.CRITICAL, force=True)

    # ------------------------------------------------------------------
    # STANDARD ALERTS
    # ------------------------------------------------------------------

    def alert_step_skipped(self, step_name: str = ""):
        msg = f"Warning. Step skipped."
        if step_name:
            msg = f"Warning. Step skipped. {step_name}"
        self.speak(msg, AlertPriority.HIGH)

    def alert_wrong_step(self, expected: str = "", detected: str = ""):
        if expected and detected:
            msg = f"Wrong step detected. Expected {expected}. Got {detected}."
        else:
            msg = "Warning. Wrong step detected."
        self.speak(msg, AlertPriority.HIGH)

    def alert_step_complete(self, step_name: str = ""):
        if step_name:
            self.speak(f"Step complete. {step_name}.", AlertPriority.LOW)
        else:
            self.speak("Step complete.", AlertPriority.LOW)

    def alert_experiment_complete(self):
        self.speak("Experiment completed successfully.", AlertPriority.NORMAL, force=True)

    def alert_calibration_required(self):
        self.speak("Camera calibration required.", AlertPriority.HIGH)

    def alert_person_not_found(self):
        self.speak("Person not detected. Please enter the camera view.", AlertPriority.NORMAL)

    def alert_step_timeout(self, step_name: str = ""):
        if step_name:
            self.speak(f"Step timeout. Please complete {step_name}.", AlertPriority.HIGH)
        else:
            self.speak("Step timeout. Please complete the current step.", AlertPriority.HIGH)

    # ------------------------------------------------------------------
    # ENABLE / DISABLE
    # ------------------------------------------------------------------

    def enable(self):
        self.voice_enabled = True
        print("[AlertManager] Voice alerts ENABLED.")

    def disable(self):
        self.voice_enabled = False
        print("[AlertManager] Voice alerts DISABLED.")

    def set_cooldown(self, seconds: float):
        self.cooldown_seconds = max(0.0, seconds)

    def reset_cooldowns(self):
        """Clear all cooldown timers (e.g., at experiment start)."""
        with self._lock:
            self._last_spoken.clear()

    # ------------------------------------------------------------------
    # INTERNAL SPEECH
    # ------------------------------------------------------------------

    def _launch_speech(self, message: str):
        """Launch speech in a daemon thread (non-blocking)."""
        # Wait for the previous thread to finish if still running
        if self._speech_thread is not None and self._speech_thread.is_alive():
            # Don't block the main thread — skip if previous is still going
            return

        self._speech_thread = threading.Thread(
            target=self._speak_worker,
            args=(message,),
            daemon=True,
            name="donuts-tts"
        )
        self._speech_thread.start()

    def _speak_worker(self, message: str):
        """Worker running on the speech thread. Creates its own pyttsx3 engine."""
        try:
            import pyttsx3
            with self._engine_lock:
                engine = pyttsx3.init()
                engine.setProperty("rate", self._voice_rate)
                engine.setProperty("volume", self._voice_volume)
                engine.say(message)
                engine.runAndWait()
                engine.stop()
        except Exception as e:
            print(f"[AlertManager] TTS error: {e}")

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def close(self):
        """Clean up resources."""
        if self._speech_thread is not None and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=2.0)
        print("[AlertManager] Closed.")
