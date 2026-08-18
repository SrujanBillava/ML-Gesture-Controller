"""
ml/data_collector.py
Interactive dataset collection tool for training the gesture classifier.

Run:
    python ml/data_collector.py

Instructions displayed on screen guide the user through capturing
landmark vectors for each gesture class.

Collected data is saved to data/raw/<gesture_name>/<timestamp>.npy
Each sample is a float32 array of shape (63,).
"""

import os
import sys
import time
import logging
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from modules.video_capture import VideoCapture
from modules.hand_detection import HandDetector
from modules.landmark_processor import LandmarkProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLES_PER_GESTURE = 1200   # capture this many samples per gesture
COUNTDOWN_SECONDS   = 3      # pause before capture starts


def collect_dataset():
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    proc     = LandmarkProcessor()
    gestures = [g for g in config.GESTURE_LABELS if g != "none"]

    with VideoCapture() as cam, HandDetector() as detector:
        fw, fh = cam.frame_size

        for gesture_name in gestures:
            gesture_dir = os.path.join(config.RAW_DATA_DIR, gesture_name)
            os.makedirs(gesture_dir, exist_ok=True)

            existing = len(os.listdir(gesture_dir))
            if existing >= SAMPLES_PER_GESTURE:
                logger.info("Skipping %s (%d samples already collected)", gesture_name, existing)
                continue

            logger.info("▶ Starting collection for: %s", gesture_name.upper())
            _countdown(cam, gesture_name, COUNTDOWN_SECONDS)

            samples_collected = existing
            target = SAMPLES_PER_GESTURE

            while samples_collected < target:
                ret, frame = cam.read()
                if not ret:
                    continue

                hands = detector.detect(frame)
                instructions = _build_instructions(
                    gesture_name, samples_collected, target
                )
                _draw_collection_ui(frame, instructions, fw, fh)

                if hands:
                    hand = hands[0]
                    feat, _ = proc.process(hand, fw, fh)
                    ts = int(time.time() * 1000)
                    path = os.path.join(gesture_dir, f"{ts}.npy")
                    np.save(path, feat)
                    samples_collected += 1
                    detector.draw_all_landmarks(frame, [hand])

                cv2.imshow("Data Collector — q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Collection aborted by user")
                    cv2.destroyAllWindows()
                    return

            logger.info("Completed: %s (%d samples)", gesture_name, target)

    cv2.destroyAllWindows()
    logger.info("Dataset collection complete.  Run ml/train_model.py to train the model.")


def _countdown(cam: VideoCapture, gesture_name: str, seconds: int):
    end = time.time() + seconds
    while time.time() < end:
        ret, frame = cam.read()
        if ret:
            remaining = int(end - time.time()) + 1
            msg = f"Get ready: {gesture_name.upper()} — starting in {remaining}s"
            cv2.putText(frame, msg, (30, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow("Data Collector — q to quit", frame)
            cv2.waitKey(1)


def _build_instructions(name: str, collected: int, total: int) -> str:
    bar_len = 30
    filled  = int(bar_len * collected / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    return f"Gesture: {name.upper()}  [{bar}] {collected}/{total}"


def _draw_collection_ui(frame, text: str, fw: int, fh: int):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    collect_dataset()
