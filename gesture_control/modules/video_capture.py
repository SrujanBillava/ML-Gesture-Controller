"""
modules/video_capture.py — Webcam Wrapper
==========================================
Responsibility: Open the webcam, deliver BGR frames, track FPS.

Why wrap OpenCV's VideoCapture?
────────────────────────────────
cv2.VideoCapture is a thin C++ binding with awkward error handling and no
built-in FPS tracking. This class adds:
  • Readable open()/release() lifecycle.
  • Context-manager support (with VideoCapture() as cam:).
  • Horizontal flip (mirror mode) so left/right match the user's perspective.
  • Rolling FPS measurement using frame timestamps.
  • Hardware buffer size = 1 to minimise latency (discard stale frames).
"""

import cv2
import logging
import time
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Thread-safe OpenCV webcam wrapper.

    Usage:
        cam = VideoCapture()
        if cam.open():
            ret, frame = cam.read()
            cam.release()

    Context-manager form:
        with VideoCapture() as cam:
            ret, frame = cam.read()
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        fps: int = config.TARGET_FPS,
    ):
        self.camera_index = camera_index
        self.width  = width
        self.height = height
        self.fps    = fps
        self._cap: Optional[cv2.VideoCapture] = None

        # _frame_times stores the monotonic timestamp of each recent frame.
        # We keep the last _fps_window frames to compute rolling FPS.
        self._frame_times: list = []
        self._fps_window = 60   # rolling window size (frames)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open and configure the camera device.

        Sets preferred resolution and FPS via OpenCV properties.
        Note: the camera may ignore these requests; use frame_size and
        measured_fps to query what was actually granted.

        Sets BUFFERSIZE = 1 to minimise latency. With the default buffer (4),
        frames from 4 frames ago can appear, introducing noticeable delay.

        Returns True on success, False if the camera couldn't be opened.
        """
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("Cannot open camera index %d", self.camera_index)
            return False

        # Request preferred settings (camera may silently override).
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        # Buffer = 1 means we always read the LATEST available frame,
        # not one that has been waiting in queue for several frames.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Log what the camera actually agreed to.
        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera opened: %dx%d @ %.1f FPS (requested %dx%d @ %d)",
            actual_w, actual_h, actual_fps,
            self.width, self.height, self.fps,
        )
        return True

    def release(self):
        """Release the camera handle and free OS resources."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released")

    def __enter__(self):
        if not self.open():
            raise RuntimeError("Failed to open camera")
        return self

    def __exit__(self, *_):
        self.release()

    # ── Frame Access ──────────────────────────────────────────────────────────

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Capture and return one frame from the webcam.

        The frame is flipped horizontally (mirror mode) so that the user's
        right hand appears on the right side of the preview window. This
        also means gesture coordinates match the user's intuitive expectation.

        Returns:
            (True,  frame_bgr) — on success
            (False, None)      — if camera is not open or frame is empty
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Empty frame received from camera")
            return False, None

        # Flip code 1 = horizontal flip (left↔right).
        frame = cv2.flip(frame, 1)

        # Record this frame's timestamp for FPS calculation.
        self._frame_times.append(time.monotonic())
        # Keep only the last N frame times to bound memory usage.
        if len(self._frame_times) > self._fps_window:
            self._frame_times.pop(0)

        return True, frame

    @property
    def measured_fps(self) -> float:
        """
        Rolling average FPS over the last N frames.

        Calculated as: (number_of_intervals) / (elapsed_time).
        Returns 0.0 if fewer than 2 frames have been captured yet.
        """
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """
        Returns (width, height) of actual captured frames.
        Queries the camera properties (reflects what the camera agreed to).
        Falls back to requested dimensions if the camera is not yet open.
        """
        if self._cap is None:
            return (self.width, self.height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def is_open(self) -> bool:
        """True if the camera device is currently open and ready."""
        return self._cap is not None and self._cap.isOpened()
