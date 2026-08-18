# 🖐️ GestureFlow

<p align="center">
  <strong>Control your computer using hand gestures through a webcam.</strong>
</p>

<p align="center">
  Real-time hand gesture recognition using Python, OpenCV, MediaPipe, and an optional machine-learning fallback.
</p>

<p align="center">
  <a href="https://github.com/SrujanBillava/ML-Gesture-Controller">GitHub Repository</a>
</p>

---

## ✨ What It Does

GestureFlow uses a webcam to detect hand landmarks and recognize hand gestures in real time. Recognized gestures are converted into computer actions such as cursor movement, mouse clicks, drag-selection, screenshots, copy, and media control.

The system uses a **hybrid gesture-recognition approach**:

* A rule-based classifier handles the primary gestures.
* An optional scikit-learn model can act as an ML fallback.
* Temporal stability filtering helps reduce single-frame misclassifications.
* Cursor movement can be smoothed to reduce jitter.
* PyAutoGUI is used to execute mouse and keyboard actions.

No special hardware is required beyond a webcam.

---

## 🚀 Features

* 🖐️ Real-time hand detection using MediaPipe
* 🎯 Cursor control using hand position
* 🖱️ Left-click using a closed-fist gesture
* 🖱️ Right-click using a pinch gesture
* ✋ Drag/select using two fingers
* 📸 Screenshot using a peace sign
* 📋 Copy using three fingers
* 👍 Play/pause media using thumbs up
* 🧠 Rule-based gesture recognition with optional ML fallback
* 📐 63-dimensional hand landmark feature representation
* 🎛️ Gesture control can be toggled using the `T` key
* ⌨️ Keyboard/mouse actions executed through PyAutoGUI

---

## 🤌 Gesture Reference

| Gesture              | Hand Position                          | Action              |
| :------------------- | :------------------------------------- | :------------------ |
| **Open Palm** 🖐️    | Flat palm facing camera                | Move cursor         |
| **Index Point** ☝️   | Only index finger extended             | Fine cursor control |
| **Closed Fist** ✊    | Fingers curled                         | Left click          |
| **Pinch** 🤏         | Thumb and index finger close together  | Right click         |
| **Two Fingers** ✌️   | Index + middle fingers close together  | Drag/select         |
| **Peace Sign** ✌️    | Index + middle fingers spread apart    | Screenshot          |
| **Thumbs Up** 👍     | Thumb extended upward                  | Play/pause          |
| **Three Fingers** 🖖 | Index + middle + ring fingers extended | Copy                |

### Keyboard Controls

| Key | Action                        |
| :-- | :---------------------------- |
| `T` | Toggle gesture control ON/OFF |
| `Q` | Quit the application          |

> The current implementation does not use a dedicated "both hands up" gesture for activation.

---

## 🏗️ Architecture

```text
                    Webcam
                       │
                       ▼
                VideoCapture
                  (OpenCV)
                       │
                       ▼
                 HandDetector
                 (MediaPipe)
                       │
                       ▼
              LandmarkProcessor
             Feature Extraction
                  63 features
                       │
                       ▼
             GestureClassifier
             ┌─────────────────┐
             │ Rule-Based      │
             │ Classification  │
             └────────┬────────┘
                      │
                 If no rule
                  matches
                      │
                      ▼
             Optional ML Model
             Random Forest /
             SVM / Gradient
             Boosting
                      │
                      ▼
             CommandExecutor
                      │
              ┌───────┴────────┐
              ▼                ▼
          PyAutoGUI       Cursor Smoothing
              │
              ▼
       Mouse / Keyboard
          Actions
```

---

## 📁 Project Structure

```text
ML-Gesture-Controller/
│
├── .gitignore
├── LICENSE
├── README.md
│
└── gesture_control/
    ├── main.py
    ├── config.py
    ├── requirements.txt
    │
    ├── modules/
    │   ├── video_capture.py
    │   ├── hand_detection.py
    │   ├── landmark_processor.py
    │   ├── gesture_classifier.py
    │   ├── cursor_smoother.py
    │   ├── command_executor.py
    │   └── ui_feedback.py
    │
    ├── ml/
    │   ├── data_collector.py
    │   ├── train_model.py
    │   └── convert.py
    │
    └── tests/
        ├── test_gesture_classifier.py
        └── test_performance.py
```

---

## 🧠 Machine Learning Pipeline

The project represents each detected hand using **63 numerical features**.

MediaPipe provides:

```text
21 hand landmarks
×
3 coordinates (x, y, z)
=
63 features
```

### Feature Processing

The landmark processor performs:

1. Extract landmark coordinates from MediaPipe.
2. Translate the landmarks relative to the wrist.
3. Scale the representation using the hand's size.
4. Flatten the landmarks into a 63-dimensional feature vector.

This makes the representation less dependent on the hand's position and scale in the camera frame.

### Models

The training pipeline can compare:

| Model                 | Configuration                                  |
| :-------------------- | :--------------------------------------------- |
| **Random Forest**     | 200 trees, balanced class weights              |
| **SVM**               | RBF kernel, `C=10`, StandardScaler             |
| **Gradient Boosting** | 150 estimators, max depth 5, learning rate 0.1 |

The models can be evaluated using stratified cross-validation and metrics such as precision, recall, and F1 score.

The trained model can then be loaded by the gesture classifier and used as an optional fallback when the rule-based engine does not recognize a gesture.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/SrujanBillava/ML-Gesture-Controller.git
cd ML-Gesture-Controller/gesture_control
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The project uses a MediaPipe version compatible with the legacy `mp.solutions.hands` API used by the current implementation.

---

## ▶️ Running the Application

From the `gesture_control` directory:

```bash
python main.py
```

The application uses the default webcam.

### Controls

Press:

```text
T → Toggle gesture control
Q → Quit
```

The rule-based gesture functionality does not require training a new ML model.

---

## 🧪 Optional ML Training

If you want to train the optional ML fallback:

### 1. Collect gesture samples

```bash
python ml/data_collector.py
```

### 2. Train and evaluate models

```bash
python ml/train_model.py
```

### 3. Run with ML fallback

```bash
python main.py --ml
```

The ML fallback uses the trained model and label encoder produced by the training pipeline.

---

## 🧪 Testing

Run the test suite from the `gesture_control` directory:

```bash
pytest tests/ -v
```

The test suite contains gesture-classification and performance-related tests.

---

## ⚙️ Configuration

Most tunable parameters are stored in:

```text
gesture_control/config.py
```

Examples include:

| Parameter                 | Purpose                                |
| :------------------------ | :------------------------------------- |
| `MP_DETECTION_CONFIDENCE` | MediaPipe hand detection confidence    |
| `MP_MODEL_COMPLEXITY`     | MediaPipe model complexity             |
| `PINCH_THRESHOLD`         | Distance threshold for pinch detection |
| `GESTURE_COOLDOWN`        | Prevents repeated discrete actions     |
| `CURSOR_SMOOTHING_ALPHA`  | Controls cursor smoothing              |
| `KALMAN_PROCESS_NOISE`    | Controls Kalman filter responsiveness  |
| `KALMAN_MEASURE_NOISE`    | Controls Kalman filter smoothing       |
| `CURSOR_DEAD_ZONE`        | Ignores very small cursor movements    |

---

## 🖥️ Requirements

### Hardware

* Webcam
* Computer capable of running Python and the required computer-vision libraries

### Software

* Python
* OpenCV
* MediaPipe
* NumPy
* scikit-learn
* PyAutoGUI
* Other dependencies listed in `requirements.txt`

The current implementation has been developed and tested on Windows.

---

## 🔧 Extending the Project

A new gesture can be added by:

1. Adding a new gesture type to the `Gesture` enum.
2. Adding detection logic to the rule-based classifier.
3. Mapping the gesture if it is used by the ML model.
4. Adding the corresponding action to `CommandExecutor`.
5. Adding or updating tests.
6. Collecting training data and retraining the ML model if necessary.

---

## ⚠️ Current Limitations

* Gesture recognition can be affected by poor lighting and difficult backgrounds.
* The primary rule-based recognition operates on a single detected hand.
* Some gestures require careful positioning to avoid overlap with other gestures.
* The optional ML fallback requires a trained model.
* System-level actions depend on the operating system and PyAutoGUI support.

---

## 👥 Development

GestureFlow was **co-developed as a team project**.

The repository contains the implementation, ML pipeline, tests, configuration, and documentation required to run and extend the project.

---

## 📝 License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  <strong>Built with Python · MediaPipe · OpenCV · scikit-learn · PyAutoGUI</strong>
</p>
