from __future__ import annotations

import logging
import time
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from modules.hand_detection import DetectedHand, LM
from modules.landmark_processor import LandmarkProcessor
import config

logger = logging.getLogger(__name__)

# ======================================================
# Gesture types
# ======================================================

class Gesture(Enum):
    NONE = auto()
    ON = auto()           # Gesture activation state
    OPEN_PALM = auto()
    PINCH = auto()
    TWO_FINGER_SCROLL = auto()
    PEACE_SIGN = auto()
    THUMBS_UP = auto()
    INDEX_POINT = auto()
    THREE_FINGER = auto()
    FIST = auto()# index + middle + ring  → Copy


@dataclass
class ClassificationResult:
    gesture: Gesture
    confidence: float
    source: str
    raw_label: str


# ======================================================
# Temporal Stability Buffer
# Requires the same gesture for N consecutive frames
# before confirming it. Eliminates single-frame
# misclassifications without adding significant latency.
# ======================================================

# Gestures that need no temporal smoothing — low-latency cursor/scroll control.
_CONTINUOUS_GESTURES = frozenset({
    Gesture.OPEN_PALM,
    Gesture.INDEX_POINT,
    Gesture.TWO_FINGER_SCROLL,
})

_RESULT_NONE = ClassificationResult(Gesture.NONE, 0.0, "stability", "none")


class GestureStabilityBuffer:
    """
    Emits a ClassificationResult only after the same Gesture appears
    in `required_frames` consecutive classifications.

    Continuous gestures (cursor/scroll) bypass this filter entirely so
    that cursor movement stays immediate and jitter-free.
    """

    def __init__(self, required_frames: int = 3):
        self._required = required_frames
        self._buf: deque = deque(maxlen=required_frames)
        self._last_confirmed: Optional[ClassificationResult] = None

    def update(self, result: ClassificationResult) -> ClassificationResult:
        # Continuous gestures are always passed through immediately
        if result.gesture in _CONTINUOUS_GESTURES:
            self._buf.clear()
            self._last_confirmed = result
            return result

        self._buf.append(result.gesture)

        # Confirm only when all N slots hold the same non-NONE gesture
        if (len(self._buf) == self._required
                and len(set(self._buf)) == 1
                and self._buf[0] != Gesture.NONE):
            self._last_confirmed = result
            return result

        return _RESULT_NONE

    def clear(self):
        self._buf.clear()


# ======================================================
# Classifier
# ======================================================

class GestureClassifier:
    def __init__(self):
        self.processor = LandmarkProcessor()
        self.ml_model = None
        self.label_encoder = None

        # Temporal filter: discrete gestures need N consecutive identical
        # frames before confirming. Continuous gestures bypass this.
        self._stability = GestureStabilityBuffer(config.GESTURE_STABILITY_FRAMES)

        logger.info("GestureClassifier initialised")

    # ==================================================

    def load_ml_model(
        self,
        model_path=config.MODEL_PATH,
        label_path=config.LABEL_PATH
    ):
        try:
            import joblib
            self.ml_model = joblib.load(model_path)
            self.label_encoder = joblib.load(label_path)
            logger.info("ML model loaded")
            return True
        except Exception as e:
            logger.warning("ML model not loaded: %s", e)
            return False

    # ==================================================

    def classify(
        self,
        hand: DetectedHand,
        frame_w,
        frame_h,
        all_hands=None
    ) -> ClassificationResult:
        """
        Classify hand gesture and return a temporally-stable result.
        Continuous gestures (cursor/scroll) bypass the stability filter.
        """
        raw = self._raw_classify(hand, frame_w, frame_h, all_hands)
        return self._stability.update(raw)

    # ==================================================

    def _raw_classify(
        self,
        hand: DetectedHand,
        frame_w,
        frame_h,
        all_hands=None
    ) -> ClassificationResult:
        """
        Raw, unfiltered classification (single frame, no temporal smoothing).
        """
        lms = hand.landmarks

        # ── Single-hand rule-based detection ─────────────────────────────────
        rule_result = self._rule_based(lms)
        if rule_result.gesture != Gesture.NONE:
            return rule_result

        # ── ML fallback ───────────────────────────────────────────────────────
        if self.ml_model:
            return self._ml_based(hand, frame_w, frame_h)

        return rule_result

    # ==================================================
    # IMPROVED RULE ENGINE
    # Changes vs original:
    #   • Uses DIP joint in addition to PIP for more robust extension check
    #   • Tighter PINCH threshold with explicit finger-curl requirement
    #   • Clearer TWO_FINGER vs PEACE_SIGN disambiguation
    # ==================================================

    def _rule_based(self, lms) -> ClassificationResult:
        # ── Finger state ──────────────────────────────────────────────────────
        # A finger is "extended" when its tip is above both its PIP *and* DIP.
        # Using DIP makes the check more robust against partially-bent fingers.
        def _extended(tip, pip, dip):
            return lms[tip].y < lms[pip].y and lms[tip].y < lms[dip].y

        index_up  = _extended(LM.INDEX_TIP,  LM.INDEX_PIP,  LM.INDEX_DIP)
        middle_up = _extended(LM.MIDDLE_TIP, LM.MIDDLE_PIP, LM.MIDDLE_DIP)
        ring_up   = _extended(LM.RING_TIP,   LM.RING_PIP,   LM.RING_DIP)
        pinky_up  = _extended(LM.PINKY_TIP,  LM.PINKY_PIP,  LM.PINKY_DIP)

        # Thumb: compare tip against IP joint (lateral motion)
        thumb_up = lms[LM.THUMB_TIP].y < lms[LM.THUMB_IP].y

        extended = sum([index_up, middle_up, ring_up, pinky_up])

        pinch_dist = LandmarkProcessor.pinch_distance(lms)

        # ── PINCH (thumb tip touching index tip) → Right Click ────────────────
        # Highest priority: if thumb and index are close, it's always a pinch
        # regardless of what the other fingers are doing.
        if pinch_dist < config.PINCH_THRESHOLD and index_up:
            return ClassificationResult(Gesture.PINCH, 0.91, "rule", "pinch")

        # ── OPEN PALM (4+ fingers extended) ─────────────────────────────────
        # Cursor control gesture.
        if extended >= 4:
            return ClassificationResult(Gesture.OPEN_PALM, 0.95, "rule", "palm")

        # ── THUMBS UP ─────────────────────────────────────────────────────────
        # Thumb tip clearly above thumb IP, all fingers curled
        if thumb_up and extended == 0:
            return ClassificationResult(Gesture.THUMBS_UP, 0.94, "rule", "thumbs_up")

        if not thumb_up and extended == 0:
            return ClassificationResult(Gesture.FIST, 0.94, "rule", "fist")

        # ── TWO FINGERS (index + middle only) ────────────────────────────────
        if index_up and middle_up and not ring_up and not pinky_up:
            dist = abs(lms[LM.INDEX_TIP].x - lms[LM.MIDDLE_TIP].x)

            # Fingers close together → scroll/drag
            if dist < 0.06:
                return ClassificationResult(
                    Gesture.TWO_FINGER_SCROLL, 0.92, "rule", "two_up"
                )
            # Fingers spread in V shape → screenshot / peace
            if dist >= 0.06:
                return ClassificationResult(
                    Gesture.PEACE_SIGN, 0.90, "rule", "peace"
                )

        # ── THREE FINGERS (index + middle + ring, no pinky) → Copy ───────────
        if index_up and middle_up and ring_up and not pinky_up:
            return ClassificationResult(
                Gesture.THREE_FINGER, 0.91, "rule", "three_finger"
            )

        # ── INDEX POINTER ─────────────────────────────────────────────────────
        if index_up and not middle_up and not ring_up and not pinky_up:
            return ClassificationResult(
                Gesture.INDEX_POINT, 0.92, "rule", "index"
            )

        return ClassificationResult(Gesture.NONE, 0, "none", "none")

    # ==================================================
    # ML FALLBACK
    # ==================================================

    def _ml_based(self, hand, fw, fh) -> ClassificationResult:
        try:
            feat, _ = self.processor.process(hand, fw, fh)
            proba = self.ml_model.predict_proba([feat])[0]
            idx = np.argmax(proba)
            label = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(proba[idx])

            if confidence < 0.60:   # raised from 0.55 for fewer false positives
                return ClassificationResult(Gesture.NONE, confidence, "ml", label)

            return ClassificationResult(
                label_to_gesture(label), confidence, "ml", label
            )
        except Exception:
            return ClassificationResult(Gesture.NONE, 0, "none", "none")

    # Backward-compat alias
    def ml_based(self, hand, fw, fh):
        return self._ml_based(hand, fw, fh)

    # Backward-compat alias (used by tests)
    def rule_based(self, lms):
        return self._rule_based(lms)

    def reset(self):
        """Reset all temporal state — call when hands disappear from frame."""
        self._stability.clear()


# ======================================================
# ML LABEL MAPPING
# ======================================================

LABEL_MAP = {
    "palm":              Gesture.OPEN_PALM,
    "open_palm":         Gesture.OPEN_PALM,
    "peace":             Gesture.PEACE_SIGN,
    "peace_sign":        Gesture.PEACE_SIGN,
    "thumbs_up":         Gesture.THUMBS_UP,
    "thumbs up":         Gesture.THUMBS_UP,
    "two_up":            Gesture.TWO_FINGER_SCROLL,
    "two_up_inverted":   Gesture.TWO_FINGER_SCROLL,
    "two_finger_scroll": Gesture.TWO_FINGER_SCROLL,
    "ok":                Gesture.PINCH,
    "pinch":             Gesture.PINCH,
    "index_point":       Gesture.INDEX_POINT,
    "three_finger":      Gesture.THREE_FINGER,
    "three_fingers":     Gesture.THREE_FINGER,
    "none":              Gesture.NONE,
    "gesture_on":        Gesture.ON,
}

def label_to_gesture(label: str) -> Gesture:
    return LABEL_MAP.get(label, Gesture.NONE)