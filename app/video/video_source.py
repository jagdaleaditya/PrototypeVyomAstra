"""
DONUTS - Video Source Module
============================
Provides a clean VideoSource abstraction supporting:
  - Webcam (index-based)
  - Local video file
  - RTSP / IP camera URL

Usage:
    src = VideoSource(0)           # webcam
    src = VideoSource("video.mp4") # file
    src = VideoSource("rtsp://...") # RTSP

    with src:
        while True:
            ret, frame, timestamp = src.read()
            if not ret:
                break
"""

import cv2
import time
import threading
from typing import Optional, Tuple, Union


class VideoSource:
    """
    Camera-agnostic video source.
    Wraps cv2.VideoCapture with:
      - graceful open/reconnect
      - frame timestamps
      - FPS tracking
      - buffer optimization for webcams
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 1,
        reconnect_delay: float = 2.0,
    ):
        """
        Args:
            source:           int for webcam index, str for file/RTSP path.
            width:            Requested frame width (hint, not guaranteed).
            height:           Requested frame height (hint, not guaranteed).
            fps:              Requested FPS (hint, not guaranteed for webcams).
            buffer_size:      cv2.CAP_PROP_BUFFERSIZE — 1 minimizes lag.
            reconnect_delay:  Seconds to wait between reconnect attempts.
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._is_open = False

        # FPS tracking
        self._fps_counter = 0
        self._fps_start = time.perf_counter()
        self._current_fps = 0.0

        # Frame count
        self.frame_count = 0
        self.dropped_frames = 0

        self.open()

    # ------------------------------------------------------------------
    # OPEN / CLOSE
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open or reopen the video source. Returns True on success."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()

            self._cap = cv2.VideoCapture(self.source)

            if not self._cap.isOpened():
                print(f"[VideoSource] ERROR: Could not open source: {self.source}")
                self._is_open = False
                return False

            # Apply settings
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Minimize buffer for webcams to reduce latency
            if isinstance(self.source, int):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            self._is_open = True
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            print(
                f"[VideoSource] Opened: {self.source} → "
                f"{actual_w}×{actual_h} @ {actual_fps:.1f} FPS"
            )
            return True

    def close(self):
        """Release the video capture."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._is_open = False
        print("[VideoSource] Closed.")

    def is_open(self) -> bool:
        return self._is_open

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[object], float]:
        """
        Read the next frame.

        Returns:
            (success, frame, timestamp)
            - success:   True if frame was read successfully.
            - frame:     BGR numpy array, or None on failure.
            - timestamp: time.perf_counter() when the frame was captured.
        """
        timestamp = time.perf_counter()

        with self._lock:
            if self._cap is None or not self._is_open:
                return False, None, timestamp

            ret, frame = self._cap.read()

        if not ret:
            self.dropped_frames += 1
            self._is_open = False
            print(f"[VideoSource] Frame read failed. Dropped: {self.dropped_frames}")
            return False, None, timestamp

        self.frame_count += 1

        # FPS tracking
        self._fps_counter += 1
        now = time.perf_counter()
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start = now

        return True, frame, timestamp

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def current_fps(self) -> float:
        """Measured frames per second over the last second."""
        return self._current_fps

    @property
    def actual_width(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # RECONNECT
    # ------------------------------------------------------------------

    def try_reconnect(self) -> bool:
        """
        Attempt to reopen the source after a disconnect.
        Returns True if reconnection succeeded.
        """
        print(f"[VideoSource] Attempting reconnect to: {self.source}")
        time.sleep(self.reconnect_delay)
        return self.open()

    # ------------------------------------------------------------------
    # CONTEXT MANAGER
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"VideoSource(source={self.source!r}, "
            f"open={self._is_open}, "
            f"fps={self._current_fps:.1f})"
        )
