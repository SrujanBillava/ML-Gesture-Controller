"""
main.py — Application Entry Point
===================================
This is the top-level script you run to start the gesture-control system.

High-level flow every frame
────────────────────────────
1.  VideoCapture  →  reads a BGR frame from the webcam (mirrored)
2.  HandDetector  →  runs MediaPipe to find hand landmarks in the frame
3.  LandmarkProcessor → normalises landmarks into a 63-feature vector
                        and computes each hand's bounding box (for drawing)
4.  GestureClassifier → classifies the dominant hand's gesture using
                        rule-based geometry + optional ML fallback.
                        Results are temporally filtered (stability buffer).
5.  CommandExecutor  → maps the confirmed gesture to a system action
                       (mouse move, click, keyboard shortcut, screenshot…)
6.  UIFeedback       → draws HUD overlays (FPS, gesture label, legend)
7.  cv2.imshow       → renders the annotated frame in a window

Keyboard shortcuts (while the window is focused)
──────────────────────────────────────────────────
  T  — toggle gesture control ON/OFF (same as BOTH_HANDS_UP gesture)
  Q  — quit the application cleanly
"""

import argparse
import logging
import sys
import time
from typing import Optional

import cv2
import pyautogui

import config
from modules.video_capture import VideoCapture
from modules.hand_detection import HandDetector
from modules.landmark_processor import LandmarkProcessor
from modules.gesture_classifier import GestureClassifier, ClassificationResult, Gesture
from modules.command_executor import CommandExecutor
from modules.ui_feedback import UIFeedback

# Disable PyAutoGUI's "move mouse to corner to abort" fail-safe.
# We handle clean shutdown ourselves via KeyboardInterrupt / 'Q' key.
pyautogui.FAILSAFE = False


def parse_args():
    """
    Parse command-line arguments.

    --ml          Load and use the trained sklearn ML model as a fallback
                  when the rule-based classifier returns NONE.
    --no-control  Run in observe-only mode: show gesture labels but do NOT
                  send any mouse/keyboard events to the OS. Useful for
                  debugging or demonstrations.
    --camera N    Select a different camera index (default from config).
    --width / --height   Override capture resolution.
    --debug       Set log level to DEBUG (very verbose, useful for tuning).
    """
    parser = argparse.ArgumentParser(
        description="Hand Gesture Based System Control"
    )
    parser.add_argument("--ml",         action="store_true")
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--camera",     type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--width",      type=int, default=config.FRAME_WIDTH)
    parser.add_argument("--height",     type=int, default=config.FRAME_HEIGHT)
    parser.add_argument("--debug",      action="store_true")
    return parser.parse_args()


def setup_logging(debug: bool):
    """
    Configure the root logger.
    All modules use logging.getLogger(__name__) so their messages inherit
    this configuration automatically.
    """
    level = logging.DEBUG if debug else getattr(logging, config.LOG_LEVEL)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-28s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    # ── 1. Bootstrap ──────────────────────────────────────────────────────────
    args = parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("main")

    # Get physical screen dimensions — used by CommandExecutor to map
    # normalised hand coordinates [0,1] → actual pixel positions.
    screen_w, screen_h = pyautogui.size()
    logger.info("Screen: %dx%d", screen_w, screen_h)

    # ── 2. Initialise all pipeline components ─────────────────────────────────
    cam        = VideoCapture(args.camera, args.width, args.height)
    detector   = HandDetector()          # wraps MediaPipe Hands
    processor  = LandmarkProcessor()    # normalises raw landmarks
    classifier = GestureClassifier()    # rule-based + optional ML

    # Optionally load the trained sklearn model for improved accuracy.
    # If the model file doesn't exist, load_ml_model() logs a warning and
    # falls back to pure rule-based classification.
    if args.ml:
        classifier.load_ml_model()

    # In --no-control mode the executor is None; gesture control is still
    # shown visually but no OS events are emitted.
    executor = None if args.no_control else CommandExecutor(screen_w, screen_h)

    # ── 3. Open camera ────────────────────────────────────────────────────────
    if not cam.open():
        logger.error("Cannot open camera")
        sys.exit(1)

    fw, fh = cam.frame_size   # actual frame width / height after camera opens

    # UIFeedback needs frame dimensions to position HUD elements correctly.
    ui = UIFeedback(fw, fh)

    logger.info("Press T to toggle gesture control")
    logger.info("Press Q to quit")

    # ── 4. Main loop state ────────────────────────────────────────────────────
    frame_count = 0
    last_result: Optional[ClassificationResult] = None   # last confirmed gesture
    last_action = ""                                      # last action string for HUD
    had_hands = False                                     # track hand presence for state reset

    # ── 5. Frame loop ─────────────────────────────────────────────────────────
    try:
        while True:
            # Read one frame from the webcam (already mirrored by VideoCapture).
            ret, frame = cam.read()
            if not ret:
                continue   # skip empty frames (e.g. camera buffer warming up)

            # Run MediaPipe to get a list of DetectedHand objects.
            # Each hand has 21 landmarks, a handedness label, and confidence.
            hands = detector.detect(frame)

            if hands:
                # Compute normalised feature vectors and bounding boxes for ALL
                # detected hands. This is a side-effectful call: it also attaches
                # the bounding box to each hand.bbox so UIFeedback can draw it.
                processor.process_batch(hands, fw, fh)

                # Choose the most-confidently detected hand as the "primary" hand.
                # This is the hand whose gesture drives all system actions.
                primary = max(hands, key=lambda h: h.confidence)

                # Classify the primary hand's gesture. all_hands is passed so
                # the classifier can detect the BOTH_HANDS_UP toggle gesture.
                result = classifier.classify(primary, fw, fh, all_hands=hands)
                last_result = result   # keep for HUD display even between frames

                # Optionally overlay MediaPipe skeleton on the frame.
                if config.SHOW_LANDMARKS:
                    detector.draw_all_landmarks(frame, hands)

                # Send the gesture to the executor which translates it into
                # OS-level actions (mouse move, click, hotkey, etc.).
                if executor:
                    last_action = executor.execute(result, primary, fw, fh)

                had_hands = True

            else:
                # No hands in frame — reset classifier and executor state so
                # stale stability-buffer entries or active drags can't persist.
                if had_hands:
                    classifier.reset()
                    if executor:
                        executor.reset()
                    had_hands = False

            # Draw all HUD overlays: FPS counter, gesture label, mode indicator,
            # action log, and the gesture legend. Returns the annotated frame.
            frame = ui.draw(
                frame,
                hands,
                last_result,
                last_action,
                cam.measured_fps,
                executor.is_active if executor else False,
            )

            # Display the annotated frame in a named window.
            cv2.imshow("Gesture Control (T=ON/OFF, Q=Quit)", frame)

            # waitKey(1) processes GUI events and returns the pressed key.
            # The & 0xFF mask strips high bits for cross-platform compatibility.
            key = cv2.waitKey(1) & 0xFF

            # T key: manually toggle gesture control (same effect as gesture).
            if key == ord("t"):
                if executor:
                    executor.toggle()
                print("CONTROL:", executor.is_active if executor else False)

            # Q key: break the loop and shut down cleanly.
            if key == ord("q"):
                break

            frame_count += 1

    except KeyboardInterrupt:
        # Ctrl+C in terminal — shut down cleanly without a scary traceback.
        logger.info("Interrupted — shutting down cleanly")

    finally:
        # The finally block ALWAYS runs (normal exit, Ctrl+C, or exception).
        # Release the webcam handle and close all OpenCV windows.
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()