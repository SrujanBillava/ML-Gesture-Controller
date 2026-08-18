"""
modules/ui_feedback.py — On-Screen HUD (Heads-Up Display)
===========================================================
Responsibility: Draw all visual overlays onto the camera frame before it
is displayed to the user. This module is purely presentational — it reads
state from other modules but never modifies it.

HUD elements drawn each frame
──────────────────────────────
  ┌────────────────────────────────────────────┐
  │ FPS: 29.8          GESTURE: ON             │  ← top bar
  │ Show BOTH HANDS to enable (when OFF)       │
  │                                            │
  │   [hand bounding box + skeleton]           │
  │                                            │
  │     OPEN PALM (95%)   ← gesture label      │  ← centre-bottom
  │ Cursor (450, 300)  ← action log entries    │  ← bottom-left
  │                    Palm → hover cursor     │  ← legend (top-right)
  │                    Fist → click            │
  └────────────────────────────────────────────┘

Anti-flicker design
────────────────────
The gesture label uses a "sticky" display: once a high-confidence result
arrives, the label is shown for 0.5 seconds even if the next frame returns
NONE. This prevents the label from blinking on every frame.
"""

from __future__ import annotations
from collections import deque
from typing import List, Optional
import time

import cv2
import numpy as np

from modules.hand_detection import DetectedHand
from modules.gesture_classifier import ClassificationResult, Gesture
import config


class UIFeedback:
    """
    Stateful HUD renderer. State it tracks between frames:
      _fps_smoothed  : exponentially smoothed FPS for a stable display value.
      _action_log    : deque of the last 5 non-trivial action strings.
      _last_gesture  : last confirmed gesture result (for anti-flicker label).
      _last_time     : when _last_gesture was last updated (for 0.5s timeout).
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h

        # Rolling FPS display (smoothed to avoid jittery numbers).
        self._fps_smoothed = 0.0

        # Circular buffer of the last 5 executed actions for the action log.
        self._action_log: deque = deque(maxlen=5)

        # Anti-flicker: remember the last high-confidence gesture so the label
        # stays on screen for 0.5s even if the next frame returns NONE.
        self._last_gesture = None
        self._last_time = time.monotonic()

    def draw(
        self,
        frame: np.ndarray,
        hands: List[DetectedHand],
        result: Optional[ClassificationResult],
        action: str,
        fps: float,
        is_active: bool,
    ) -> np.ndarray:
        """
        Draw all HUD elements onto `frame` and return the annotated frame.

        frame      : BGR image from VideoCapture (modified in-place)
        hands      : list of DetectedHand objects from this frame
        result     : latest ClassificationResult (may be None if no hand)
        action     : human-readable string from CommandExecutor
        fps        : measured_fps from VideoCapture
        is_active  : whether gesture control is currently ON
        """
        # ── FPS smoothing ─────────────────────────────────────────────────────
        # EMA with alpha=0.15: the display value changes slowly to avoid
        # jumping between e.g. "29" and "31" every frame.
        self._fps_smoothed = 0.85 * self._fps_smoothed + 0.15 * fps

        # ── Bounding boxes (optional) ─────────────────────────────────────────
        # Draw a rectangle and confidence label around each detected hand.
        # hand.bbox is set by LandmarkProcessor.process_batch() in main.py.
        if config.SHOW_LANDMARKS:
            for hand in hands:
                self._draw_bbox(frame, hand)

        # ── FPS counter (top-left) ────────────────────────────────────────────
        if config.SHOW_FPS:
            self._draw_fps(frame)

        # ── Mode indicator (top-right) ────────────────────────────────────────
        # Shows "GESTURE: ON" in green or "GESTURE: OFF" in grey.
        if config.SHOW_MODE_INDICATOR:
            self._draw_mode(frame, is_active)

        # ── Gesture label (anti-flicker, centre-bottom) ───────────────────────
        # Only update the sticky label when a CONFIDENT non-NONE result arrives.
        # The label then stays visible for 0.5 seconds after the last update.
        if config.SHOW_GESTURE_LABEL and result and result.gesture != Gesture.NONE:
            if result.confidence > 0.65:
                self._last_gesture = result
                self._last_time = time.monotonic()   # reset the 0.5s window

        # Show the sticky gesture as long as we're within the 0.5-second window.
        if self._last_gesture and time.monotonic() - self._last_time < 0.5:
            self._draw_gesture_label(frame, self._last_gesture)

        # ── Action log (bottom-left) ──────────────────────────────────────────
        # Append non-trivial action strings (suppress "Idle" and "OFF" noise).
        if action and action not in ("Idle", "Gesture control OFF"):
            self._action_log.appendleft(action)

        self._draw_action_log(frame)

        # ── Gesture legend (top-right panel) ─────────────────────────────────
        self._draw_legend(frame, is_active)

        return frame

    # ── Private drawing helpers ───────────────────────────────────────────────

    def _draw_bbox(self, frame, hand):
        """
        Draw a coloured rectangle around the hand plus a confidence label.
        Uses hand.bbox (x, y, w, h) set earlier by LandmarkProcessor.
        """
        if hand.bbox is None:
            return

        x, y, w, h = hand.bbox
        colour = config.ACTIVE_COLOUR

        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

        # Confidence shown as percentage e.g. "Right 94%"
        label = f"{hand.handedness} {hand.confidence:.0%}"
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    def _draw_fps(self, frame):
        """Draw smoothed FPS in the top-left corner."""
        text = f"FPS: {self._fps_smoothed:.1f}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    config.LANDMARK_COLOUR, 2)

    def _draw_mode(self, frame, is_active):
        """
        Draw the GESTURE ON/OFF indicator in the top-right corner.
        Green when active, grey when inactive.
        """
        colour = config.ACTIVE_COLOUR if is_active else config.INACTIVE_COLOUR
        label  = "GESTURE: ON" if is_active else "GESTURE: OFF"
        cv2.putText(frame, label, (self.fw - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    def _draw_gesture_label(self, frame, result):
        """
        Draw the confirmed gesture name and confidence in the lower-centre.
        e.g. "OPEN PALM (95%)"
        """
        text = f"{result.raw_label.upper()} ({result.confidence:.0%})"
        cv2.putText(frame, text,
                    (self.fw // 2 - 150, self.fh - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    config.LABEL_TEXT_COLOUR,
                    2)

    def _draw_action_log(self, frame):
        """
        Draw the last 5 executed action strings stacked above the bottom edge.
        The most recent action appears at the bottom; older entries go upward.
        """
        for i, entry in enumerate(self._action_log):
            cv2.putText(frame, entry,
                        (10, self.fh - 10 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (180, 200, 180),
                        1)

    def _draw_legend(self, frame, is_active):
        """
        Draw the gesture cheat-sheet in the top-right corner.

        When gesture control is OFF: show the activation hint.
        When ON: show two columns of gesture → action mappings.
          Left column  (green)  : cursor/mouse gestures
          Right column (blue)   : clipboard/media gestures
        """
        if not is_active:
            # Prompt the user to activate gesture control.
            cv2.putText(frame, "Press T to enable",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (150, 150, 150),
                        1)
            return

        # Left column — cursor / mouse
        left_col = [
            "Palm   → hover cursor",
            "Pinch  → right-click",
            "2 fin  → drag-select",
        ]
        # Right column — clipboard / media (in blue to distinguish)
        right_col = [
            "3 fin  → copy  (C-c)",
            "Peace  → screenshot",
            "Thumb  → play/pause",
        ]

        for i, line in enumerate(left_col):
            cv2.putText(frame, line,
                        (self.fw - 260, 50 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (180, 200, 180),
                        1)

        for i, line in enumerate(right_col):
            cv2.putText(frame, line,
                        (self.fw - 260, 160 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (150, 210, 255),
                        1)