"""
config.py — Central Configuration File
=======================================
All tunable parameters for the entire gesture-control system live here.
Changing a value here affects every module that imports config, so you
never need to hunt through multiple files to adjust behaviour.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
# BASE_DIR points to the gesture_control/ folder (where this file lives).
# Every other path is built relative to BASE_DIR so the project is portable.
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")          # training/processed data
RAW_DATA_DIR  = os.path.join(DATA_DIR, "raw")           # raw .npy landmark files
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")     # (reserved for future use)
MODELS_DIR    = os.path.join(BASE_DIR, "models")        # saved model artefacts
LOGS_DIR      = os.path.join(BASE_DIR, "logs")          # log files (if file handler added)
MODEL_PATH    = os.path.join(MODELS_DIR, "gesture_classifier.pkl")   # sklearn pipeline
LABEL_PATH    = os.path.join(MODELS_DIR, "label_encoder.pkl")        # LabelEncoder

# ─── Camera ───────────────────────────────────────────────────────────────────
# CAMERA_INDEX: 0 = first webcam, 1 = second, etc.
# Lower resolution gives better FPS and more stable MediaPipe tracking.
CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
TARGET_FPS    = 30

# ─── MediaPipe Hand Detection ─────────────────────────────────────────────────
# MediaPipe runs a two-stage pipeline:
#   1. Palm detection (expensive) — fires when tracking is lost
#   2. Landmark tracking (cheap) — runs every frame once a hand is found
#
# MP_MAX_HANDS: set to 2 so the BOTH_HANDS_UP toggle gesture works.
# Detection/tracking confidence: lower values = more permissive but noisier.
# Model complexity 0 = lite model (faster, slightly less accurate).
MP_MAX_HANDS            = 2
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE  = 0.5
MP_MODEL_COMPLEXITY     = 0

# ─── Gesture Classification ───────────────────────────────────────────────────
# PINCH_THRESHOLD: maximum normalised distance between thumb tip and index tip
# that counts as a "pinch". Smaller = stricter pinch required.
PINCH_THRESHOLD = 0.07

# Temporal stability — how many consecutive identical frames must be seen
# before a gesture is confirmed. This eliminates single-frame glitches.
#   • GESTURE_STABILITY_FRAMES: for all one-shot gestures (click, copy, etc.)
#   • TOGGLE_STABILITY_FRAMES : stricter window for the mode-toggle gesture
# Continuous gestures (cursor, scroll) bypass these filters entirely.
GESTURE_STABILITY_FRAMES = 3    # ≈ 100 ms at 30 fps

# Cooldowns (seconds) — minimum time between consecutive executions of a gesture.
# Prevents a single held gesture from firing the action many times.
GESTURE_COOLDOWN    = 0.6    # default for one-shot gestures (click, copy…)
SCREENSHOT_COOLDOWN = 2.5    # screenshot: avoid burst-saving many files

# ─── Cursor Control ───────────────────────────────────────────────────────────
# CURSOR_SMOOTHING_ALPHA: EMA alpha (0–1). Higher = faster but jitterier.
# KALMAN_PROCESS_NOISE (Q): how much the filter trusts hand motion.
# KALMAN_MEASURE_NOISE (R): how much the filter trusts MediaPipe landmarks.
# CURSOR_DEAD_ZONE: ignore movements smaller than this (reduces micro-jitter).
# SCREEN_MARGIN_FRAC: fraction of screen edge kept as a buffer zone so the
#   cursor is still reachable at the extremes without extreme hand positions.
CURSOR_SMOOTHING_ALPHA = 0.4
KALMAN_PROCESS_NOISE   = 5e-3
KALMAN_MEASURE_NOISE   = 2e-1
CURSOR_DEAD_ZONE       = 0.01
SCREEN_MARGIN_FRAC     = 0.05

# ─── Drag / Select ────────────────────────────────────────────────────────────
DRAG_COOLDOWN = 0.0   # continuous gesture — no cooldown needed

# ─── UI Overlay ───────────────────────────────────────────────────────────────
# These booleans let you toggle individual HUD elements on/off.
SHOW_LANDMARKS     = True    # draw hand skeleton + bounding box
SHOW_GESTURE_LABEL = True    # show confirmed gesture name in frame centre
SHOW_FPS           = True    # show rolling FPS counter top-left
SHOW_MODE_INDICATOR = True   # show GESTURE ON / OFF indicator top-right

# OpenCV colour tuples are (Blue, Green, Red) — opposite of standard RGB.
LANDMARK_COLOUR   = (0, 255, 128)    # bright green — landmark dots & skeleton
CONNECTION_COLOUR = (128, 255, 0)    # yellow-green — skeleton connection lines
LABEL_BG_COLOUR   = (30, 30, 30)     # dark grey — background behind gesture label
LABEL_TEXT_COLOUR = (255, 255, 255)  # white — gesture label text
ACTIVE_COLOUR     = (0, 220, 80)     # green  — shown when gesture control is ON
INACTIVE_COLOUR   = (60, 60, 60)     # grey   — shown when gesture control is OFF

# ─── Gesture Labels ───────────────────────────────────────────────────────────
# Master list used by the data collector and training pipeline.
# The string names here match the folder names under data/raw/<gesture>/.
# Adding a new gesture: add its name here AND handle it in gesture_classifier.py.
GESTURE_LABELS = [
    "open_palm",
    "pinch",
    "two_finger_scroll",
    "thumbs_up",
    "peace_sign",
    "index_point",
    "three_finger",   # → Ctrl+C (copy)
    "none",
]

# ─── ML Model Hyperparameters ────────────────────────────────────────────────
# FEATURE_SIZE = 21 landmarks × 3 coords (x, y, z) = 63 features per sample.
NUM_LANDMARKS  = 21
LANDMARK_DIMS  = 3
FEATURE_SIZE   = NUM_LANDMARKS * LANDMARK_DIMS   # = 63

ML_TEST_SIZE    = 0.2    # 20% of data reserved for held-out evaluation
ML_RANDOM_STATE = 42     # fixed seed for reproducible train/test splits
ML_CV_FOLDS     = 5      # k-folds for cross-validation

# ─── Logging ──────────────────────────────────────────────────────────────────
# LOG_LEVEL controls the console verbosity. Use "DEBUG" to see per-frame detail.
LOG_LEVEL        = "INFO"
LOG_PERFORMANCE  = True