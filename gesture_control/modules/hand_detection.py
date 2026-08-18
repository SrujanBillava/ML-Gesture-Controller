"""
modules/hand_detection.py — MediaPipe Hand Detector Wrapper
=============================================================
Responsibility: Convert a raw BGR camera frame into a list of
DetectedHand objects, each containing 21 3-D landmarks.

Why wrap MediaPipe?
───────────────────
MediaPipe's raw API returns landmark objects that are tightly coupled
to its own types. By converting them to plain Python dataclasses
(Landmark, DetectedHand) we decouple the rest of the system from
MediaPipe's internals, making unit testing and future backend swaps easy.

Landmark coordinate system
───────────────────────────
MediaPipe returns NORMALISED coordinates in [0, 1]:
  x = 0 → left edge of frame,    x = 1 → right edge
  y = 0 → top edge of frame,     y = 1 → bottom edge
  z ≈ depth relative to wrist (negative = closer to camera)
Y increases downward. Keep this in mind when checking "finger is above joint".
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark Index Mapping
# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe defines 21 landmarks numbered 0–20.
# The LM class gives each index a human-readable name so the rest of the code
# can write lms[LM.INDEX_TIP] instead of the magic number lms[8].
#
# Hand anatomy reference (simplified):
#   CMC = Carpometacarpal joint (base, connects to wrist)
#   MCP = Metacarpophalangeal joint (knuckle)
#   IP  = Interphalangeal joint (thumb only)
#   PIP = Proximal Interphalangeal joint (middle knuckle)
#   DIP = Distal Interphalangeal joint (upper knuckle, just below fingertip)
#   TIP = Fingertip
#
# Visual layout (palm facing camera):
#
#          PINKY  RING  MIDDLE  INDEX    THUMB
#   TIP     20    16     12      8        4
#   DIP     19    15     11      7        3 (IP for thumb)
#   PIP     18    14     10      6        2 (MCP for thumb)
#   MCP     17    13      9      5        1 (CMC for thumb)
#   WRIST   ─────────── 0 ───────────────
# ─────────────────────────────────────────────────────────────────────────────

class LM:
    WRIST = 0

    # Thumb — note the different joint naming (no PIP/DIP, uses IP/CMC/MCP)
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP  = 3
    THUMB_TIP = 4

    # Index finger
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    # Middle finger
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    # Ring finger
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    # Pinky finger
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Landmark:
    """
    One of the 21 MediaPipe hand landmarks.
    Coordinates are normalised to [0, 1] relative to the frame dimensions.
    z is depth relative to the wrist (approximate, not metric).
    """
    x: float
    y: float
    z: float


@dataclass
class DetectedHand:
    """
    All data associated with a single detected hand.

    landmarks   : list of 21 Landmark objects (LM.WRIST … LM.PINKY_TIP)
    handedness  : "Left" or "Right" as seen from the camera's perspective
                  (note: mirrored frames flip this label)
    confidence  : classification confidence from MediaPipe (0–1)
    bbox        : (x, y, w, h) bounding box in pixel coords, set later by
                  LandmarkProcessor.process(). None until that call is made.
    """
    landmarks: List[Landmark]
    handedness: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Hand Detector
# ─────────────────────────────────────────────────────────────────────────────

class HandDetector:
    """
    Wraps MediaPipe's Hands solution to detect and track hands in video frames.

    Usage:
        detector = HandDetector()
        hands = detector.detect(bgr_frame)   # returns List[DetectedHand]

    Context-manager form (used by data_collector.py):
        with HandDetector() as detector:
            hands = detector.detect(frame)
    """

    def __init__(
        self,
        max_hands: int = config.MP_MAX_HANDS,
        detection_confidence: float = config.MP_DETECTION_CONFIDENCE,
        tracking_confidence: float = config.MP_TRACKING_CONFIDENCE,
    ):
        self.mp_hands = mp.solutions.hands

        # MediaPipe Hands configuration:
        #   static_image_mode=False  → tracking mode (faster than detection every frame)
        #   max_num_hands            → how many hands to find simultaneously
        #   model_complexity=0       → lite model; =1 for full model (more accurate, slower)
        #   min_detection_confidence → how confident MP must be to initially detect a hand
        #   min_tracking_confidence  → how confident MP must be to keep tracking an existing hand
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=config.MP_MODEL_COMPLEXITY,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.mp_draw = mp.solutions.drawing_utils
        logger.info("HandDetector initialised")

    # ── Context manager support (for data_collector.py) ───────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, frame) -> List[DetectedHand]:
        """
        Run MediaPipe on one BGR frame and return all detected hands.

        Steps:
          1. Convert BGR → RGB (MediaPipe expects RGB input)
          2. Run MediaPipe Hands.process() — returns normalised landmarks
          3. Convert each hand's landmarks + handedness into DetectedHand objects

        Returns an empty list if no hands are found.
        """
        # MediaPipe requires RGB; OpenCV provides BGR by default.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process() is where the neural network inference happens.
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []   # no hands detected in this frame

        detected = []
        # MediaPipe returns two parallel lists: one for landmarks, one for
        # handedness classification. zip() pairs them up correctly.
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            # Convert MediaPipe's landmark objects to our plain Landmark dataclass.
            landmarks = [
                Landmark(lm.x, lm.y, lm.z)
                for lm in hand_landmarks.landmark
            ]

            detected.append(DetectedHand(
                landmarks=landmarks,
                # classification[0] is the top-1 prediction for handedness
                handedness=handedness.classification[0].label,
                confidence=handedness.classification[0].score,
            ))

        return detected

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_all_landmarks(self, frame, hands):
        """
        Draw hand skeleton directly onto the frame (in-place).

        Does NOT re-run inference — it uses the already-detected landmark
        coordinates stored in each DetectedHand object.

        Process:
          1. Convert normalised coords → pixel coords using frame dimensions.
          2. Draw connection lines between landmark pairs defined by
             MediaPipe's HAND_CONNECTIONS constant.
          3. Draw filled circles at each landmark point.
        """
        h_px, w_px = frame.shape[:2]
        for hand in hands:
            # Convert normalised [0,1] to pixel [0, w_px] / [0, h_px]
            lm_coords = [
                (int(l.x * w_px), int(l.y * h_px))
                for l in hand.landmarks
            ]
            # Draw skeleton connections (bones)
            for connection in self.mp_hands.HAND_CONNECTIONS:
                pt1 = lm_coords[connection[0]]
                pt2 = lm_coords[connection[1]]
                cv2.line(frame, pt1, pt2, (128, 255, 0), 2)
            # Draw landmark dots (joints)
            for pt in lm_coords:
                cv2.circle(frame, pt, 4, (0, 255, 128), -1)
        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()