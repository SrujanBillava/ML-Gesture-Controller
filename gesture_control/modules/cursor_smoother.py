"""
modules/cursor_smoother.py — Cursor Smoothing Strategies
==========================================================
Responsibility: Reduce jitter in the cursor position by filtering the raw,
noisy hand landmark coordinates before mapping them to screen pixels.

Why is smoothing needed?
────────────────────────
MediaPipe landmark coordinates flicker by a few pixels every frame even when
the hand is perfectly still (camera sensor noise + neural network uncertainty).
Without smoothing, the cursor would shake constantly, making precise control
impossible.

Two strategies are provided:
  1. ExponentialSmoother — simple, O(1) per frame, very low latency.
  2. KalmanSmoother      — optimal Bayesian filter, handles sudden movements
                           better; recommended for production use.

CursorMapper wraps a smoother and adds:
  • Dead zone — ignores tiny movements below a threshold.
  • Margin clamping — keeps cursor away from screen edges.
  • Coordinate remapping — stretches the usable hand area to fill the screen.
"""

from __future__ import annotations
import numpy as np
import config


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Exponential Moving Average (EMA)
# ─────────────────────────────────────────────────────────────────────────────

class ExponentialSmoother:
    """
    Applies an exponential moving average to the cursor position.

    Formula each frame:
        smooth = alpha * raw + (1 - alpha) * prev_smooth

    With alpha = 1.0 the smoother is a pass-through (no smoothing).
    With alpha = 0.1 the smoother reacts very slowly to changes (very smooth
    but feels laggy). A value around 0.3–0.5 is a good balance.

    The first frame seeds the smoother with the raw value (no lag on start).
    """

    def __init__(self, alpha: float = config.CURSOR_SMOOTHING_ALPHA):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        # None until the first update() call.
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

    def update(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """Apply EMA and return the smoothed (x, y)."""
        if self._smooth_x is None:
            # First frame: initialise directly with raw position (no delay).
            self._smooth_x, self._smooth_y = raw_x, raw_y
            return raw_x, raw_y

        self._smooth_x = self.alpha * raw_x + (1 - self.alpha) * self._smooth_x
        self._smooth_y = self.alpha * raw_y + (1 - self.alpha) * self._smooth_y
        return self._smooth_x, self._smooth_y

    def reset(self):
        """Forget smoothing history — called when gesture mode is toggled."""
        self._smooth_x = self._smooth_y = None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — 1-D Kalman Filter (applied to X and Y independently)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanSmoother:
    """
    A simple scalar Kalman filter applied independently to X and Y.

    Kalman Filter Concepts
    ──────────────────────
    The filter maintains a STATE ESTIMATE (where we think the hand is)
    and an ERROR COVARIANCE (how confident we are).

    Each frame:
      PREDICT — project the state forward using the motion model
                (constant position: no velocity assumed).
      UPDATE  — correct the prediction using the new measurement (landmark).
                The correction is weighted by the Kalman Gain K:
                  K = P_pred / (P_pred + R)
                  x_new = x_pred + K * (measurement - x_pred)

    Tuning
    ──────
    Q (process_noise)  : How much does the hand actually move between frames?
                         Increase if the cursor feels sluggish/laggy.
    R (measure_noise)  : How noisy are the MediaPipe measurements?
                         Increase for smoother output; decrease for faster response.
    """

    def __init__(
        self,
        process_noise: float = config.KALMAN_PROCESS_NOISE,
        measure_noise: float = config.KALMAN_MEASURE_NOISE,
    ):
        self.Q = process_noise   # Q: process noise covariance
        self.R = measure_noise   # R: measurement noise covariance

        # State estimates for X and Y axes (both start at 0; seeded on first call).
        self._x_hat = np.array([0.0])   # estimated position (x-axis)
        self._y_hat = np.array([0.0])   # estimated position (y-axis)

        # Error covariances — start high (uncertain) and converge over frames.
        self._Px = np.array([1.0])
        self._Py = np.array([1.0])

        self._initialised = False

    def update(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """Run one Kalman filter step and return smoothed (x, y)."""
        if not self._initialised:
            # Seed with the first measurement so cursor jumps to hand immediately.
            self._x_hat[0] = raw_x
            self._y_hat[0] = raw_y
            self._initialised = True
            return raw_x, raw_y

        # Run the predict + update cycle for both axes.
        sx = self._kalman_step(self._x_hat, self._Px, raw_x)
        sy = self._kalman_step(self._y_hat, self._Py, raw_y)

        # Unpack (new_position, new_covariance) tuples.
        self._x_hat[0], self._Px[0] = sx
        self._y_hat[0], self._Py[0] = sy

        return float(self._x_hat[0]), float(self._y_hat[0])

    def _kalman_step(
        self, x_hat: np.ndarray, P: np.ndarray, measurement: float
    ) -> tuple[float, float]:
        """
        One predict + update step for a single axis.

        Returns (new_position_estimate, new_error_covariance).
        """
        # PREDICT: assume constant position (no velocity model).
        x_pred = x_hat[0]
        P_pred = P[0] + self.Q   # uncertainty grows without measurement

        # UPDATE: compute Kalman gain and correct the prediction.
        K = P_pred / (P_pred + self.R)   # gain: 0 = trust model, 1 = trust sensor
        x_new = x_pred + K * (measurement - x_pred)   # weighted correction
        P_new = (1 - K) * P_pred                       # reduce uncertainty

        return x_new, P_new

    def reset(self):
        """Reset filter state — called when gesture mode is toggled."""
        self._initialised = False
        self._Px = np.array([1.0])
        self._Py = np.array([1.0])


# ─────────────────────────────────────────────────────────────────────────────
# CursorMapper — ties smoothing, dead-zone, and screen mapping together
# ─────────────────────────────────────────────────────────────────────────────

class CursorMapper:
    """
    Maps normalised hand coordinates [0, 1] to screen pixel coordinates.

    Pipeline each frame:
      1. Dead-zone check   — if the hand moved less than CURSOR_DEAD_ZONE,
                             use the previous position to ignore micro-jitter.
      2. Smooth            — run the Kalman filter to reduce larger jitter.
      3. Margin clamp      — keep the cursor inside [m, 1-m] so the hand
                             doesn't need to go to the extreme edge to reach
                             screen corners.
      4. Remap             — stretch the clamped [m, 1-m] range to [0, screen_dim].

    Usage:
        mapper = CursorMapper(screen_w=1920, screen_h=1080)
        screen_x, screen_y = mapper.map(norm_x, norm_y)
    """

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._prev_norm = None   # previous normalised (x, y) for dead-zone check

        # KalmanSmoother is used by default. Swap with ExponentialSmoother
        # if you prefer simpler tuning at the cost of less optimal filtering.
        self._smoother = KalmanSmoother()

    def map(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        """
        Map normalised [0,1] frame coordinates to screen pixel coordinates.

        norm_x, norm_y : MediaPipe wrist or fingertip position in [0,1]
        returns        : (screen_x, screen_y) integer pixel coordinates
        """
        # ── 1. Dead zone ──────────────────────────────────────────────────────
        # If the hand movement since the last frame is smaller than the dead-zone
        # threshold, freeze the cursor at its current position to suppress jitter.
        if self._prev_norm is not None:
            dx = abs(norm_x - self._prev_norm[0])
            dy = abs(norm_y - self._prev_norm[1])
            if dx < config.CURSOR_DEAD_ZONE and dy < config.CURSOR_DEAD_ZONE:
                norm_x, norm_y = self._prev_norm   # freeze position

        self._prev_norm = (norm_x, norm_y)

        # ── 2. Smooth ─────────────────────────────────────────────────────────
        sx, sy = self._smoother.update(norm_x, norm_y)

        # ── 3. Margin clamp ───────────────────────────────────────────────────
        # Keep cursor in [m, 1-m] so the user doesn't have to move their hand
        # to the extreme edges of the camera frame to reach screen corners.
        m  = config.SCREEN_MARGIN_FRAC
        sx = np.clip(sx, m, 1 - m)
        sy = np.clip(sy, m, 1 - m)

        # ── 4. Remap [m, 1-m] → [0, screen_dim] ──────────────────────────────
        # Linear stretch: the usable hand range fills the full screen.
        screen_x = int((sx - m) / (1 - 2 * m) * self.screen_w)
        screen_y = int((sy - m) / (1 - 2 * m) * self.screen_h)

        # Final clamp to valid pixel range (safety net).
        screen_x = int(np.clip(screen_x, 0, self.screen_w - 1))
        screen_y = int(np.clip(screen_y, 0, self.screen_h - 1))

        return screen_x, screen_y

    def reset(self):
        """Reset smoother and dead-zone history (called on mode toggle)."""
        self._smoother.reset()
        self._prev_norm = None
