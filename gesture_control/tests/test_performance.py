"""
tests/test_performance.py
Performance and latency benchmarks.
Run with:  pytest tests/test_performance.py -v -s
"""

import sys, os, time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.landmark_processor import LandmarkProcessor
from modules.gesture_classifier import GestureClassifier
from modules.cursor_smoother import KalmanSmoother, ExponentialSmoother, CursorMapper
from modules.hand_detection import DetectedHand, Landmark, LM


def _rand_hand():
    lms = [Landmark(np.random.rand(), np.random.rand(), np.random.rand() * 0.1)
           for _ in range(21)]
    return DetectedHand(landmarks=lms, handedness="Right", confidence=0.9)


class TestClassifierLatency:

    def test_classifier_latency_under_150ms(self):
        """
        Full classification pipeline must complete in < 150ms per frame
        (requirement: <150ms end-to-end latency).
        We test the pure compute path excluding I/O.
        """
        clf  = GestureClassifier()
        proc = LandmarkProcessor()
        N    = 300

        hand = _rand_hand()
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            proc.process(hand, 1280, 720)
            clf.classify(hand, 1280, 720)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        p99_ms = np.percentile(times, 99)
        print(f"\nClassifier avg: {avg_ms:.2f} ms   p99: {p99_ms:.2f} ms")
        assert p99_ms < 150, f"p99 latency {p99_ms:.1f}ms exceeds 150ms budget"

    def test_feature_extraction_speed(self):
        proc = LandmarkProcessor()
        hand = _rand_hand()
        N    = 1000

        t0 = time.perf_counter()
        for _ in range(N):
            proc.process(hand, 1280, 720)
        elapsed = (time.perf_counter() - t0) * 1000 / N

        print(f"\nFeature extraction avg: {elapsed:.3f} ms")
        assert elapsed < 5.0, f"Feature extraction too slow: {elapsed:.2f} ms"


class TestKalmanSmoother:

    def test_smoother_converges(self):
        smoother = KalmanSmoother()
        # Feed constant position — should converge to that position
        for _ in range(50):
            sx, sy = smoother.update(0.5, 0.5)
        assert abs(sx - 0.5) < 0.01
        assert abs(sy - 0.5) < 0.01

    def test_smoother_reduces_jitter(self):
        """Smoothed signal variance must be lower than raw signal variance."""
        smoother = KalmanSmoother()
        np.random.seed(42)
        true_pos = 0.5
        raw_x    = true_pos + np.random.normal(0, 0.05, 200)
        smooth_x = [smoother.update(x, 0.5)[0] for x in raw_x]

        assert np.var(smooth_x) < np.var(raw_x), "Smoother must reduce variance"

    def test_exponential_smoother_alpha_bounds(self):
        with pytest.raises(ValueError):
            ExponentialSmoother(alpha=0.0)
        with pytest.raises(ValueError):
            ExponentialSmoother(alpha=1.1)


class TestCursorMapper:

    def test_cursor_within_screen_bounds(self):
        mapper = CursorMapper(1920, 1080)
        for _ in range(100):
            x, y = mapper.map(np.random.rand(), np.random.rand())
            assert 0 <= x < 1920, f"x={x} out of bounds"
            assert 0 <= y < 1080, f"y={y} out of bounds"

    def test_cursor_maps_centre_correctly(self):
        mapper = CursorMapper(1920, 1080)
        # Centre of frame should map near centre of screen (after warm-up)
        for _ in range(30):
            x, y = mapper.map(0.5, 0.5)
        assert abs(x - 960) < 150, f"Centre x={x} too far from 960"
        assert abs(y - 540) < 150, f"Centre y={y} too far from 540"
