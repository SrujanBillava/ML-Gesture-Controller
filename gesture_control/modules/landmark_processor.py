"""
modules/landmark_processor.py — Landmark Normalisation & Feature Extraction
=============================================================================
Responsibility: Convert a raw DetectedHand (21 normalised MediaPipe landmarks)
into a scale- and translation-invariant 63-dimensional feature vector that
can be fed directly to a scikit-learn classifier.

Why normalise?
──────────────
MediaPipe coordinates depend on WHERE the hand is in the frame and HOW FAR
it is from the camera. Without normalisation:
  • A close-up fist and a distant fist look like completely different inputs.
  • A hand in the top-left corner has different raw coords than the same
    hand in the bottom-right corner.

After normalisation:
  • All feature vectors represent the SHAPE of the hand, not its position/size.
  • The same gesture performed at any distance/position produces ~the same vector.

Normalisation pipeline (for each hand)
────────────────────────────────────────
  1. Scale to pixels  → multiply [0,1] coords by frame width/height
  2. Compute bbox     → min/max pixel bounding box of all 21 landmarks
  3. Translate        → subtract wrist position so wrist = (0, 0, 0)
  4. Scale            → divide by bounding-box diagonal so shape fits in [0,1]
  5. Flatten          → reshape (21, 3) → (63,) float32 array

This gives a feature vector that is invariant to translation and scale,
but still encodes the finger angles (which define the gesture).
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from modules.hand_detection import DetectedHand, Landmark, LM
import config


class LandmarkProcessor:
    """
    Stateless processor: converts one DetectedHand into a feature vector.

    All geometry helper methods (finger_extended, pinch_distance, etc.) are
    @staticmethods so the rule-based classifier can call them without needing
    a processor instance.
    """

    def process(
        self,
        hand: DetectedHand,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Normalise one hand and return (feature_vector, bbox).

        feature_vector : float32 array of shape (63,)
                         21 landmarks × [x, y, z] after wrist-centering and
                         diagonal scaling.
        bbox           : (x, y, w, h) in PIXEL coordinates, clamped to frame.
                         Also stored in hand.bbox as a side-effect so UIFeedback
                         can draw the bounding box without re-running this method.
        """
        lms = hand.landmarks

        # ── Step 1: Convert normalised → pixel coordinates ────────────────────
        # MediaPipe gives x,y in [0,1]. Multiply by frame dimensions to get
        # actual pixel positions. z is kept as-is (relative depth, not pixels).
        px = np.array([[lm.x * frame_w, lm.y * frame_h, lm.z] for lm in lms])

        # ── Step 2: Bounding box ──────────────────────────────────────────────
        # Find the min/max pixel extents of all landmarks, add padding,
        # and clamp so the box never extends outside the frame.
        x_min, y_min = px[:, 0].min(), px[:, 1].min()
        x_max, y_max = px[:, 0].max(), px[:, 1].max()
        pad = 20   # pixels of padding around the hand
        bx  = max(0, int(x_min) - pad)
        by  = max(0, int(y_min) - pad)
        bx2 = min(frame_w, int(x_max) + pad)
        by2 = min(frame_h, int(y_max) + pad)
        bbox = (bx, by, bx2 - bx, by2 - by)   # (x, y, width, height)

        # ── Step 3: Translate — wrist to origin ───────────────────────────────
        # Subtract the wrist's pixel position from every landmark so the wrist
        # ends up at (0, 0, 0). This makes the feature vector position-invariant.
        wrist    = px[LM.WRIST].copy()
        px_norm  = px - wrist

        # ── Step 4: Scale — bounding box diagonal = 1.0 ──────────────────────
        # Divide by the diagonal length of the hand bounding box so the
        # feature vector is scale-invariant (same gesture near/far = same vector).
        diag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        if diag > 0:
            px_norm /= diag

        # ── Step 5: Flatten to 1-D ───────────────────────────────────────────
        # Reshape (21, 3) → (63,) and cast to float32 (sklearn/numpy standard).
        feature_vector = px_norm.flatten().astype(np.float32)

        # Store bbox on the hand object so UI modules can draw it without
        # needing to re-run this computation.
        hand.bbox = bbox

        return feature_vector, bbox

    def process_batch(
        self,
        hands: List[DetectedHand],
        frame_w: int,
        frame_h: int,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Process multiple hands; returns a list of (feature_vector, bbox)."""
        return [self.process(h, frame_w, frame_h) for h in hands]

    # ── Geometry Helper Methods ───────────────────────────────────────────────
    # These are used by the rule-based gesture classifier to compute
    # geometric properties of the hand from raw (unnormalised) landmarks.
    # They are @staticmethods so they can be called without instantiation.

    @staticmethod
    def finger_extended(lms: List[Landmark], finger_tip: int, finger_pip: int) -> bool:
        """
        Returns True when a finger appears "extended" (pointing upward/outward).

        Logic: the fingertip must be HIGHER in the frame than the PIP joint.
        Because y increases downward, "higher" means smaller y value.
        So: tip.y < pip.y  →  tip is above pip  →  finger is extended.

        Note: this is a simple single-joint check. The classifier uses the
        DIP joint as well for better robustness (see _rule_based in classifier).
        """
        return lms[finger_tip].y < lms[finger_pip].y

    @staticmethod
    def all_four_fingers_extended(lms: List[Landmark]) -> bool:
        """True when index, middle, ring, AND pinky are all extended."""
        tips_pips = [
            (LM.INDEX_TIP,  LM.INDEX_PIP),
            (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
            (LM.RING_TIP,   LM.RING_PIP),
            (LM.PINKY_TIP,  LM.PINKY_PIP),
        ]
        return all(LandmarkProcessor.finger_extended(lms, t, p) for t, p in tips_pips)

    @staticmethod
    def all_fingers_curled(lms: List[Landmark]) -> bool:
        """
        True when all four fingertips are below (larger y than) their MCP joints.
        Used as a stricter "fist" check that considers the knuckle position.
        """
        tips_mcps = [
            (LM.INDEX_TIP,  LM.INDEX_MCP),
            (LM.MIDDLE_TIP, LM.MIDDLE_MCP),
            (LM.RING_TIP,   LM.RING_MCP),
            (LM.PINKY_TIP,  LM.PINKY_MCP),
        ]
        return all(lms[tip].y > lms[mcp].y for tip, mcp in tips_mcps)

    @staticmethod
    def euclidean_distance(a: Landmark, b: Landmark) -> float:
        """3-D Euclidean distance between two landmarks (in normalised coords)."""
        return float(np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2))

    @staticmethod
    def pinch_distance(lms: List[Landmark]) -> float:
        """
        Distance between thumb tip and index tip in normalised coordinates.
        Values < config.PINCH_THRESHOLD (0.07) indicate a pinch/OK gesture.
        """
        return LandmarkProcessor.euclidean_distance(
            lms[LM.THUMB_TIP], lms[LM.INDEX_TIP]
        )

    @staticmethod
    def hand_rotation_angle(lms: List[Landmark]) -> float:
        """
        Angle of the hand's "spine" (wrist → middle MCP) in degrees.

        This angle changes as the user tilts/rotates their hand, making it
        ideal for mapping to continuous controls like volume adjustment.
        Returns a value in [-180, 180] degrees via arctan2.
        """
        dx = lms[LM.MIDDLE_MCP].x - lms[LM.WRIST].x
        dy = lms[LM.MIDDLE_MCP].y - lms[LM.WRIST].y
        return float(np.degrees(np.arctan2(dy, dx)))

    @staticmethod
    def index_tip_coords(lms: List[Landmark]) -> Tuple[float, float]:
        """Return (x, y) of the index fingertip in normalised [0,1] coords."""
        return lms[LM.INDEX_TIP].x, lms[LM.INDEX_TIP].y

    @staticmethod
    def wrist_coords(lms: List[Landmark]) -> Tuple[float, float]:
        """Return (x, y) of the wrist in normalised [0,1] coords."""
        return lms[LM.WRIST].x, lms[LM.WRIST].y

    @staticmethod
    def count_extended_fingers(lms: List[Landmark], include_thumb: bool = False) -> int:
        """
        Count how many of the four fingers (index, middle, ring, pinky)
        are currently extended.

        The thumb is excluded by default because its extension axis is
        lateral (side-to-side) rather than vertical, so tip.y < pip.y
        doesn't work reliably for the thumb.

        Set include_thumb=True only when you specifically need the thumb
        counted and have accounted for its different geometry.
        """
        fingers = [
            (LM.INDEX_TIP,  LM.INDEX_PIP),
            (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
            (LM.RING_TIP,   LM.RING_PIP),
            (LM.PINKY_TIP,  LM.PINKY_PIP),
        ]
        count = sum(1 for t, p in fingers if lms[t].y < lms[p].y)
        if include_thumb and lms[LM.THUMB_TIP].y < lms[LM.THUMB_MCP].y:
            count += 1
        return count
