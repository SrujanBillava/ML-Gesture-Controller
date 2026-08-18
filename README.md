<p align="center">
  <h1 align="center">🖐️ GestureFlow</h1>
  <p align="center">
    <strong>Control your entire computer with just your hands — no mouse, no keyboard, no special hardware.</strong>
  </p>
  <p align="center">
    Real-time hand gesture recognition using webcam + MediaPipe + ML &middot; Built with Python
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-gesture-reference">Gestures</a> •
    <a href="#%EF%B8%8F-architecture">Architecture</a> •
    <a href="#-ml-pipeline">ML Pipeline</a> •
    <a href="#-configuration">Config</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10-00897B?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## ✨ What It Does

GestureFlow turns your webcam into a full system controller. Point, click, drag, scroll, adjust volume, take screenshots — all with natural hand gestures. No gloves, sensors, or special hardware needed.

### Highlights

- 🎯 **12 gesture classes** — cursor, click, drag-select, volume, media, navigation, screenshots
- 🧠 **Hybrid engine** — deterministic rule-based classifier + ML fallback (RandomForest / SVM / GradientBoosting)
- 📐 **Kalman-filtered cursor** — smooth, low-jitter cursor movement with dead-zone support
- ⚡ **Real-time** — 25–30 FPS on an i3 CPU with a 720p webcam, end-to-end latency < 150ms
- 🖥️ **Cross-platform** — Windows, macOS, Linux

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/GestureFlow.git
cd GestureFlow/gesture_control

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run (works immediately — no training needed)

```bash
python main.py
```

> **To activate:** Show **both hands** to the camera — or press **T**.
> **To quit:** Press **Q**.

### 3. (Optional) Train the ML model for improved accuracy

```bash
# Step 1: Collect gesture samples (guided UI)
python ml/data_collector.py

# Step 2: Train — best model is auto-saved
python ml/train_model.py

# Step 3: Run with ML fallback enabled
python main.py --ml
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## 🤌 Gesture Reference

| Gesture | Your Hand | What It Does |
|:--------|:----------|:-------------|
| **Open Palm** 🖐️ | Flat palm facing camera | Move cursor (hover mode) |
| **Index Point** ☝️ | Only index finger up | Fine cursor control |
| **Closed Fist** ✊ | All fingers curled | **Left click** / open app |
| **Pinch** 🤏 | Thumb + index finger together | **Right click** |
| **Two Fingers** ✌️ | Index + middle up (close together) | **Drag & select** text |
| **Peace Sign** ✌️ | Index + middle up (spread apart) | **Screenshot** (saved to ~/Pictures) |
| **Thumbs Up** 👍 | Thumb up, fingers curled | **Play / Pause** media |
| **Swipe Left** 👈 | Quick hand movement left | **Browser back** |
| **Swipe Right** 👉 | Quick hand movement right | **Browser forward** |
| **Hand Rotation** 🔄 | Tilt hand left/right | **Volume up / down** |
| **Both Hands Up** 🙌 | Raise both hands | **Toggle gesture control ON/OFF** |

### Workflow Example

```
1. Show both hands → gesture control activates
2. Open palm → cursor follows your hand (hover)
3. Move over an app icon
4. Close fist → click! App opens
5. Two fingers up → drag to select text
6. Release to any other gesture → selection complete
```

---

## 🏗️ Architecture

```
  ┌──────────────────────────────────────────────────────────────┐
  │                        MAIN LOOP                             │
  │                                                              │
  │  Webcam ──► VideoCapture ──► HandDetector ──► LandmarkProc.  │
  │              (OpenCV)        (MediaPipe)     (Normalise 63D) │
  │                                    │                         │
  │                                    ▼                         │
  │                          GestureClassifier                   │
  │                          ┌──────────────┐                    │
  │                          │ Rule Engine  │ ◄── Fast, primary  │
  │                          │ ML Fallback  │ ◄── RF/SVM/GB      │
  │                          └──────┬───────┘                    │
  │                                 │                            │
  │                                 ▼                            │
  │                          CommandExecutor                     │
  │                          ┌──────────────┐                    │
  │                          │ CursorMapper │ + Kalman filter    │
  │                          │ PyAutoGUI    │ mouse/keyboard/etc │
  │                          │ State Mgmt   │ cooldowns, drag    │
  │                          └──────┬───────┘                    │
  │                                 │                            │
  │                                 ▼                            │
  │                          UIFeedback (HUD overlay)            │
  │                          Landmarks │ Gesture │ FPS │ Mode    │
  └──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
gesture_control/
├── main.py                     # Entry point — run this
├── config.py                   # All tuneable parameters in one place
├── requirements.txt
│
├── modules/
│   ├── video_capture.py        # Webcam wrapper with FPS tracking
│   ├── hand_detection.py       # MediaPipe Hands + landmark dataclasses
│   ├── landmark_processor.py   # Normalisation, feature extraction, geometry
│   ├── gesture_classifier.py   # Hybrid rule-based + ML gesture engine
│   ├── cursor_smoother.py      # Kalman filter + exponential smoother
│   ├── command_executor.py     # System actions (click, drag, volume, etc.)
│   └── ui_feedback.py          # Real-time HUD overlay
│
├── ml/
│   ├── data_collector.py       # Interactive dataset collection UI
│   ├── train_model.py          # Full training + evaluation pipeline
│   └── convert.py              # Image dataset → .npy landmark vectors
│
├── tests/
│   ├── test_gesture_classifier.py   # Unit tests (15 tests)
│   └── test_performance.py          # Latency benchmarks (7 tests)
│
├── data/raw/                   # Per-gesture .npy landmark samples
├── models/                     # Saved model + label encoder (.pkl)
└── logs/                       # Runtime logs
```

---

## 🧠 ML Pipeline

### Feature Engineering

Each hand → **63 normalised features** (21 landmarks × [x, y, z]):
1. Extract pixel coordinates from MediaPipe landmarks
2. Translate so wrist = origin (position-invariant)
3. Scale by bounding box diagonal (size-invariant)
4. Flatten to 1D float32 vector

### Models Compared

| Model | Details |
|-------|---------|
| **RandomForest** | 200 trees, balanced class weights, no scaler needed |
| **SVM (RBF)** | C=10, gamma=scale, StandardScaler preprocessing |
| **GradientBoosting** | 150 estimators, max_depth=5, lr=0.1 |

All evaluated with **5-fold stratified cross-validation**. Best model (by weighted F1) is auto-saved.

### Evaluation Output
- Per-class precision / recall / F1 report
- Confusion matrix heatmap (saved to `models/`)
- Train/test accuracy comparison

---

## ⚙️ Configuration

All parameters live in [`config.py`](gesture_control/config.py):

| Parameter | Default | What It Controls |
|-----------|:-------:|------------------|
| `MP_DETECTION_CONFIDENCE` | 0.5 | Hand detection sensitivity (lower = more detections) |
| `MP_MODEL_COMPLEXITY` | 0 | MediaPipe model (0=lite/fast, 1=full/accurate) |
| `PINCH_THRESHOLD` | 0.07 | Max thumb-index distance to register a pinch |
| `GESTURE_COOLDOWN` | 0.6s | Debounce for discrete gestures (click, play/pause) |
| `CURSOR_SMOOTHING_ALPHA` | 0.4 | EMA factor (lower=smoother, higher=more responsive) |
| `KALMAN_PROCESS_NOISE` | 5e-3 | Kalman filter agility (higher=faster tracking) |
| `KALMAN_MEASURE_NOISE` | 2e-1 | Kalman filter smoothness (higher=smoother) |
| `CURSOR_DEAD_ZONE` | 0.01 | Ignore movements smaller than this |
| `SWIPE_MIN_DISPLACEMENT` | 0.18 | Min wrist travel to trigger a swipe |

---

## 🖥️ Hardware Requirements

| | Minimum | Recommended |
|:--|:--------|:------------|
| **CPU** | Intel i3 / Ryzen 3 | Intel i5+ / Ryzen 5+ |
| **RAM** | 4 GB | 8 GB |
| **Webcam** | 720p @ 30 FPS | 1080p @ 30 FPS |
| **OS** | Windows 10 / Ubuntu 20.04 / macOS 11 | Any recent version |

---

## 🛠️ CLI Options

```bash
python main.py [OPTIONS]

  --ml          Enable ML model fallback (requires trained model)
  --no-control  View-only mode (no system actions)
  --camera N    Camera index (default: 0)
  --width W     Frame width (default: 640)
  --height H    Frame height (default: 480)
  --debug       Verbose logging
```

---

## 🔧 Extending the Project

### Add a new gesture
1. Add the name to `config.GESTURE_LABELS`
2. Add an enum value in `gesture_classifier.Gesture`
3. Add detection logic in `GestureClassifier.rule_based()`
4. Map the label in `LABEL_MAP` (for ML)
5. Add the action in `CommandExecutor.execute()`
6. Collect data & retrain if using ML

### Swap the cursor smoother
In `cursor_smoother.CursorMapper.__init__()`, replace:
```python
self._smoother = KalmanSmoother()        # current
self._smoother = ExponentialSmoother()   # alternative
```

### Use a deep learning model
Replace the sklearn pipeline in `train_model.py` with a CNN accepting (21, 3) landmark arrays. The classifier's `ml_based()` method just calls `model.predict()` — swap the model and it works transparently.

---

## ⚠️ Known Limitations

- Accuracy degrades in very low light (< 50 lux)
- Glossy / reflective backgrounds can confuse MediaPipe
- Single-hand primary recognition (multi-hand only for toggle)
- Volume control uses OS keyboard shortcuts (requires media key support)
- Hand rotation gesture requires ML model (no rule-based detection)

---

## 📝 License

MIT — free to use, modify, and distribute.

---

<p align="center">
  <strong>Built with 🐍 Python &middot; 🖐️ MediaPipe &middot; 👁️ OpenCV &middot; 🤖 scikit-learn</strong>
</p>
