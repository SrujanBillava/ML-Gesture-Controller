"""
modules/command_executor.py
Maps ClassificationResult → system-level actions.
Uses PyAutoGUI for cross-platform cursor/keyboard/screenshot control.
Maintains gesture state (active mode, cooldowns, scroll velocity).
"""

from __future__ import annotations
import logging
import platform
import time
from typing import Optional, Tuple

import pyautogui as pag

from modules.gesture_classifier import Gesture, ClassificationResult
from modules.cursor_smoother import CursorMapper
from modules.hand_detection import DetectedHand, LM
from modules.landmark_processor import LandmarkProcessor
import config

logger = logging.getLogger(__name__)

# PyAutoGUI inter-call sleep disabled — timing is handled by our cooldown logic
pag.PAUSE = 0.0


class GestureMode:
    """Tracks whether gesture control is globally active."""
    INACTIVE = "inactive"
    ACTIVE   = "active"


class CommandExecutor:
    """
    Stateful command dispatcher.

    Call execute(result, hand, frame_w, frame_h) each frame.
    The executor manages:
      - Cursor mode toggle and cursor movement
      - Click debouncing
      - Scroll velocity
      - Volume adjustment via keyboard
      - Swipe navigation
      - Screenshot capture
      - Mode toggle (both hands up)
    """

    def __init__(self, screen_w: int, screen_h: int):
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._mapper = CursorMapper(screen_w, screen_h)
        self._mode = GestureMode.INACTIVE

        # Cooldown tracking
        self._last_gesture_time: dict[Gesture, float] = {}

        # Cursor mode state
        self._cursor_active = False

        # Drag/select state (two-finger gesture)
        self._drag_active = False

        logger.info("CommandExecutor ready: screen=%dx%d", screen_w, screen_h)

    # ── Main Dispatch ─────────────────────────────────────────────────────────

    def execute(
        self,
        result: ClassificationResult,
        hand: DetectedHand,
        frame_w: int,
        frame_h: int,
    ) -> str:
        """
        Execute system command for the given ClassificationResult.
        Returns a human-readable action string for UI display.
        """
        g = result.gesture
        now = time.monotonic()

        if not self._cursor_active:
            return "Gesture control OFF"

        # ── Reset continuous-gesture state when gesture changes ────────────
        if g != Gesture.TWO_FINGER_SCROLL:
            if self._drag_active:
                pag.mouseUp()
                self._drag_active = False
                logger.debug("Drag released")


        # ── Check cooldown (skip for continuous gestures with 0.0 cooldown)
        cooldown = self._cooldown_for(g)
        if cooldown > 0.0:
            last = self._last_gesture_time.get(g, 0.0)
            if now - last < cooldown:
                return f"[cooldown] {g.name}"
        self._last_gesture_time[g] = now

        # ── Dispatch ──────────────────────────────────────────────────────────
        lms = hand.landmarks

        if g == Gesture.OPEN_PALM or g == Gesture.INDEX_POINT:
            return self._move_cursor(lms, frame_w, frame_h, g)

        elif g == Gesture.PINCH:
            pag.rightClick()
            logger.debug("Right click")
            return "Right click"

        elif g == Gesture.FIST:
            pag.click()
            logger.debug("Left click")
            return "Left click"

        elif g == Gesture.TWO_FINGER_SCROLL:
            return self._drag_select(lms, frame_w, frame_h)

        elif g == Gesture.THUMBS_UP:
            pag.press("space")  # play/pause in most media players
            return "Play / Pause"

        elif g == Gesture.PEACE_SIGN:
            self._take_screenshot()
            return "Screenshot saved"

        elif g == Gesture.THREE_FINGER:
            pag.hotkey("ctrl", "c")
            logger.debug("Copy (Ctrl+C)")
            return "Copy (Ctrl+C)"

        return "Idle"

    # ── Sub-actions ───────────────────────────────────────────────────────────

    def _move_cursor(self, lms, frame_w: int, frame_h: int, gesture: Gesture) -> str:
        """Map index fingertip to screen coordinates and move the cursor."""
        if gesture == Gesture.OPEN_PALM:
            norm_x, norm_y = LandmarkProcessor.wrist_coords(lms)
        else:
            norm_x, norm_y = LandmarkProcessor.index_tip_coords(lms)

        screen_x, screen_y = self._mapper.map(norm_x, norm_y)
        pag.moveTo(screen_x, screen_y, _pause=False)
        return f"Cursor ({screen_x}, {screen_y})"

    def _drag_select(self, lms, frame_w: int, frame_h: int) -> str:
        """Click-and-drag using index finger position. Enables text selection."""
        norm_x, norm_y = LandmarkProcessor.index_tip_coords(lms)
        screen_x, screen_y = self._mapper.map(norm_x, norm_y)

        if not self._drag_active:
            # Start drag: move to position first, then press mouse down
            pag.moveTo(screen_x, screen_y, _pause=False)
            pag.mouseDown()
            self._drag_active = True
            logger.debug("Drag started at (%d, %d)", screen_x, screen_y)
            return f"Select start ({screen_x}, {screen_y})"

        # Continue dragging
        pag.moveTo(screen_x, screen_y, _pause=False)
        return f"Selecting ({screen_x}, {screen_y})"

    def _take_screenshot(self):
        import datetime, os
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            os.path.expanduser("~"), "Pictures", f"gesture_shot_{ts}.png"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        screenshot = pag.screenshot()
        screenshot.save(path)
        logger.info("Screenshot saved: %s", path)

    # ── Cooldown Map ──────────────────────────────────────────────────────────

    @staticmethod
    def _cooldown_for(g: Gesture) -> float:
        if g == Gesture.PEACE_SIGN:
            return config.SCREENSHOT_COOLDOWN
        if g in (Gesture.OPEN_PALM, Gesture.INDEX_POINT,
                  Gesture.TWO_FINGER_SCROLL):
            return 0.0   # continuous gestures — no cooldown
        return config.GESTURE_COOLDOWN

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_active(self) -> bool:
        return self._cursor_active

    def toggle(self):
        """Toggle gesture control on/off externally (e.g. from keyboard)."""
        self.reset()
        self._cursor_active = not self._cursor_active
        self._mode = GestureMode.ACTIVE if self._cursor_active else GestureMode.INACTIVE
        logger.info("Gesture mode toggled: %s", self._mode)

    def reset(self):
        """Reset all hardware and software state. Release mouse buttons."""
        if self._drag_active:
            pag.mouseUp()
            self._drag_active = False
            logger.debug("Drag released on reset")
        self._mapper.reset()

