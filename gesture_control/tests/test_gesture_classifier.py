"""
tests/test_gesture_classifier.py
Unit tests for LandmarkProcessor and GestureClassifier.
Run with:  pytest tests/ -v --tb=short
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from modules.hand_detection import DetectedHand, Landmark, LM
from modules.landmark_processor import LandmarkProcessor
from modules.gesture_classifier import GestureClassifier, Gesture


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _make_landmarks(overrides: dict = None) -> list[Landmark]:
    """
    Build 21 landmarks all at (0.5, 0.5, 0).
    Override specific indices via a dict: {LM.INDEX_TIP: (0.5, 0.3, 0), ...}
    """
    lms = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    if overrides:
        for idx, (x, y, z) in overrides.items():
            lms[idx] = Landmark(x, y, z)
    return lms


def _hand(lms) -> DetectedHand:
    return DetectedHand(landmarks=lms, handedness="Right", confidence=0.95)


FRAME_W, FRAME_H = 1280, 720


# ── LandmarkProcessor tests ───────────────────────────────────────────────────

class TestLandmarkProcessor:

    def test_feature_vector_shape(self):
        proc = LandmarkProcessor()
        lms  = _make_landmarks()
        hand = _hand(lms)
        feat, bbox = proc.process(hand, FRAME_W, FRAME_H)
        assert feat.shape == (63,), "Feature vector must be 63-dimensional"
        assert feat.dtype == np.float32

    def test_wrist_at_origin_after_normalisation(self):
        """After normalisation the wrist landmark should be at (0, 0, 0)."""
        proc = LandmarkProcessor()
        # Wrist at different position
        lms  = _make_landmarks({LM.WRIST: (0.3, 0.7, 0.0)})
        hand = _hand(lms)
        feat, _ = proc.process(hand, FRAME_W, FRAME_H)
        wrist_feat = feat[LM.WRIST * 3: LM.WRIST * 3 + 3]
        np.testing.assert_allclose(wrist_feat, [0, 0, 0], atol=1e-5,
                                   err_msg="Wrist must be at origin")

    def test_bbox_is_inside_frame(self):
        proc = LandmarkProcessor()
        lms  = _make_landmarks()
        hand = _hand(lms)
        _, (x, y, w, h) = proc.process(hand, FRAME_W, FRAME_H)
        assert x >= 0 and y >= 0
        assert x + w <= FRAME_W
        assert y + h <= FRAME_H

    def test_finger_extended_detection(self):
        # Index tip above PIP → extended
        lms = _make_landmarks(
            {LM.INDEX_TIP: (0.5, 0.2, 0), LM.INDEX_PIP: (0.5, 0.4, 0)}
        )
        assert LandmarkProcessor.finger_extended(lms, LM.INDEX_TIP, LM.INDEX_PIP)

    def test_finger_curled_detection(self):
        # Index tip below PIP → curled
        lms = _make_landmarks(
            {LM.INDEX_TIP: (0.5, 0.7, 0), LM.INDEX_PIP: (0.5, 0.4, 0)}
        )
        assert not LandmarkProcessor.finger_extended(lms, LM.INDEX_TIP, LM.INDEX_PIP)

    def test_pinch_distance_close(self):
        lms = _make_landmarks(
            {LM.THUMB_TIP: (0.5, 0.5, 0), LM.INDEX_TIP: (0.501, 0.501, 0)}
        )
        dist = LandmarkProcessor.pinch_distance(lms)
        assert dist < 0.01

    def test_count_extended_fingers(self):
        # All four fingers extended
        lms = _make_landmarks({
            LM.INDEX_TIP:  (0.5, 0.2, 0), LM.INDEX_PIP:  (0.5, 0.4, 0),
            LM.MIDDLE_TIP: (0.5, 0.2, 0), LM.MIDDLE_PIP: (0.5, 0.4, 0),
            LM.RING_TIP:   (0.5, 0.2, 0), LM.RING_PIP:   (0.5, 0.4, 0),
            LM.PINKY_TIP:  (0.5, 0.2, 0), LM.PINKY_PIP:  (0.5, 0.4, 0),
        })
        assert LandmarkProcessor.count_extended_fingers(lms) == 4


# ── GestureClassifier tests ───────────────────────────────────────────────────

class TestGestureClassifier:

    def setup_method(self):
        self.clf = GestureClassifier()

    def _classify(self, lms):
        """
        Simulate holding a gesture for GESTURE_STABILITY_FRAMES frames.
        The stability buffer requires N consecutive identical results before
        confirming, so a single call would always return NONE for discrete
        gestures.  Continuous gestures (OPEN_PALM, INDEX_POINT …) pass
        through on frame 1 and are not affected by this.
        """
        hand = _hand(lms)
        result = None
        for _ in range(config.GESTURE_STABILITY_FRAMES):
            result = self.clf.classify(hand, FRAME_W, FRAME_H)
        return result

    def test_open_palm_detected(self):
        """All four fingers + thumb extended → OPEN_PALM."""
        lms = _make_landmarks({
            LM.INDEX_TIP:  (0.5, 0.2, 0), LM.INDEX_PIP:  (0.5, 0.4, 0),
            LM.MIDDLE_TIP: (0.5, 0.2, 0), LM.MIDDLE_PIP: (0.5, 0.4, 0),
            LM.RING_TIP:   (0.5, 0.2, 0), LM.RING_PIP:   (0.5, 0.4, 0),
            LM.PINKY_TIP:  (0.5, 0.2, 0), LM.PINKY_PIP:  (0.5, 0.4, 0),
            LM.THUMB_TIP:  (0.3, 0.2, 0), LM.THUMB_IP:   (0.35, 0.4, 0),
        })
        result = self._classify(lms)
        assert result.gesture == Gesture.OPEN_PALM

    def test_pinch_detected(self):
        """Thumb tip very close to index tip → PINCH."""
        lms = _make_landmarks({
            LM.THUMB_TIP: (0.50, 0.50, 0),
            LM.INDEX_TIP: (0.51, 0.51, 0),
            LM.MIDDLE_TIP: (0.5, 0.2, 0), LM.MIDDLE_PIP: (0.5, 0.4, 0),
            LM.RING_TIP:   (0.5, 0.2, 0), LM.RING_PIP:   (0.5, 0.4, 0),
        })
        result = self._classify(lms)
        assert result.gesture == Gesture.PINCH

    def test_peace_sign_detected(self):
        """Index + middle extended and spread apart, ring + pinky curled → PEACE_SIGN."""
        lms = _make_landmarks({
            LM.INDEX_TIP:  (0.4, 0.2, 0), LM.INDEX_PIP:  (0.45, 0.4, 0),
            LM.MIDDLE_TIP: (0.6, 0.2, 0), LM.MIDDLE_PIP: (0.55, 0.4, 0),
            LM.RING_TIP:   (0.5, 0.7, 0), LM.RING_PIP:   (0.5, 0.5, 0),
            LM.PINKY_TIP:  (0.5, 0.7, 0), LM.PINKY_PIP:  (0.5, 0.5, 0),
        })
        result = self._classify(lms)
        assert result.gesture == Gesture.PEACE_SIGN

    def test_thumbs_up_detected(self):
        """Thumb tip above MCP, all fingers curled → THUMBS_UP."""
        lms = _make_landmarks({
            LM.THUMB_TIP:  (0.5, 0.2, 0), LM.THUMB_MCP:  (0.5, 0.5, 0),
            LM.INDEX_TIP:  (0.5, 0.7, 0), LM.INDEX_MCP:  (0.5, 0.5, 0),
            LM.MIDDLE_TIP: (0.5, 0.7, 0), LM.MIDDLE_MCP: (0.5, 0.5, 0),
            LM.RING_TIP:   (0.5, 0.7, 0), LM.RING_MCP:   (0.5, 0.5, 0),
            LM.PINKY_TIP:  (0.5, 0.7, 0), LM.PINKY_MCP:  (0.5, 0.5, 0),
            LM.INDEX_PIP:  (0.5, 0.6, 0),
            LM.MIDDLE_PIP: (0.5, 0.6, 0),
            LM.RING_PIP:   (0.5, 0.6, 0),
            LM.PINKY_PIP:  (0.5, 0.6, 0),
        })
        result = self._classify(lms)
        assert result.gesture == Gesture.THUMBS_UP

    def test_index_point_detected(self):
        """Only index extended → INDEX_POINT."""
        lms = _make_landmarks({
            LM.INDEX_TIP:  (0.5, 0.2, 0), LM.INDEX_PIP:  (0.5, 0.4, 0),
            LM.MIDDLE_TIP: (0.5, 0.7, 0), LM.MIDDLE_PIP: (0.5, 0.5, 0),
            LM.RING_TIP:   (0.5, 0.7, 0), LM.RING_PIP:   (0.5, 0.5, 0),
            LM.PINKY_TIP:  (0.5, 0.7, 0), LM.PINKY_PIP:  (0.5, 0.5, 0),
        })
        result = self._classify(lms)
        assert result.gesture == Gesture.INDEX_POINT

    def test_classifier_returns_result_object(self):
        lms    = _make_landmarks()
        result = self._classify(lms)
        assert hasattr(result, "gesture")
        assert hasattr(result, "confidence")
        assert hasattr(result, "source")
        assert 0.0 <= result.confidence <= 1.0
