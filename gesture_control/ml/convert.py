"""
ml/convert.py
Converts a folder of gesture images (dataset/<gesture>/*.jpg) into
normalised .npy landmark vectors in data/raw/<gesture>/.

Uses LandmarkProcessor for feature extraction, ensuring the same
normalisation as data_collector.py and the runtime classifier.

Run:
    python ml/convert.py
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# ── Path setup (resolve relative to this script) ─────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")

sys.path.insert(0, PROJECT_DIR)

import config
from modules.hand_detection import HandDetector, DetectedHand, Landmark
from modules.landmark_processor import LandmarkProcessor

DATASET_PATH = os.path.join(PROJECT_DIR, "dataset")
OUTPUT_PATH = config.RAW_DATA_DIR

processor = LandmarkProcessor()

os.makedirs(OUTPUT_PATH, exist_ok=True)

for gesture in os.listdir(DATASET_PATH):

    input_folder = os.path.join(DATASET_PATH, gesture)

    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(OUTPUT_PATH, gesture)

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    skipped = 0

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    for img_name in os.listdir(input_folder):

        path = os.path.join(input_folder, img_name)

        img = cv2.imread(path)

        if img is None:
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            skipped += 1
            continue

        hand_landmarks = result.multi_hand_landmarks[0]

        # Build a DetectedHand so we can use LandmarkProcessor
        landmarks = [
            Landmark(lm.x, lm.y, lm.z)
            for lm in hand_landmarks.landmark
        ]
        hand = DetectedHand(
            landmarks=landmarks,
            handedness="Right",
            confidence=1.0,
        )

        # Normalise with the same pipeline as data_collector.py
        feat, _ = processor.process(hand, w, h)

        np.save(
            os.path.join(output_folder, f"{count}.npy"),
            feat
        )

        count += 1

    hands.close()
    print(f"{gesture}: converted {count}, skipped {skipped}")