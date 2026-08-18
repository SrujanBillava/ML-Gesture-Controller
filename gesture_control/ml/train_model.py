"""
ml/train_model.py
Full training pipeline for the gesture classifier.

Steps:
  1. Load raw landmark vectors from data/raw/<gesture_name>/*.npy
  2. Balance classes (optional)
  3. Train RandomForest + SVM + compare
  4. Evaluate with accuracy / precision / recall / F1 / confusion matrix
  5. Save best model to models/gesture_classifier.pkl

Run:
    python ml/train_model.py
"""

import os
import sys
import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_dataset(raw_dir: str = config.RAW_DATA_DIR):
    """
    Walk raw_dir/<gesture>/*.npy and return (X, y) arrays.
    X: float32 (N, 63),  y: string labels (N,)
    """
    X_parts, y_parts = [], []

    for gesture_name in sorted(os.listdir(raw_dir)):
        gesture_path = os.path.join(raw_dir, gesture_name)
        if not os.path.isdir(gesture_path):
            continue
        files = [f for f in os.listdir(gesture_path) if f.endswith(".npy")]
        if not files:
            logger.warning("No samples for gesture: %s", gesture_name)
            continue
        vectors = [np.load(os.path.join(gesture_path, f)) for f in files]
        X_parts.append(np.stack(vectors))
        y_parts.extend([gesture_name] * len(vectors))
        logger.info("  %-22s  %d samples", gesture_name, len(vectors))

    if not X_parts:
        raise FileNotFoundError(
            "No data found in %s — run ml/data_collector.py first." % raw_dir
        )

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.array(y_parts)
    logger.info("Dataset: %d samples, %d gestures, %d features", len(y), len(set(y)), X.shape[1])
    return X, y


def balance_classes(X, y, max_per_class: int = 1000):
    """Undersample over-represented classes for balanced training."""
    X_out, y_out = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = np.random.choice(idx, max_per_class, replace=False)
        X_out.append(X[idx])
        y_out.append(y[idx])
    return np.concatenate(X_out), np.concatenate(y_out)


# ── Model Definitions ─────────────────────────────────────────────────────────

def build_models():
    return {
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                random_state=config.ML_RANDOM_STATE,
                class_weight="balanced",
            ))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                random_state=config.ML_RANDOM_STATE,
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=config.ML_RANDOM_STATE,
            ))
        ]),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}   Weighted F1 : {f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=label_names))

    return acc, f1, confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(cm, labels, model_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Confusion matrix saved: %s", save_path)
    plt.show()


# ── Cross-Validation ──────────────────────────────────────────────────────────

def cross_validate(name, model, X, y):
    skf = StratifiedKFold(n_splits=config.ML_CV_FOLDS, shuffle=True,
                          random_state=config.ML_RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    logger.info(
        "[CV] %s — mean: %.4f  std: %.4f  [%s]",
        name, scores.mean(), scores.std(),
        ", ".join(f"{s:.3f}" for s in scores)
    )
    return scores.mean()


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, label_encoder):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(label_encoder, config.LABEL_PATH)
    logger.info("Model saved: %s", config.MODEL_PATH)
    logger.info("Label encoder saved: %s", config.LABEL_PATH)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def train():
    logger.info("Loading dataset…")
    X, y = load_dataset()
    X, y = balance_classes(X, y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_names = le.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=config.ML_TEST_SIZE,
        stratify=y_enc,
        random_state=config.ML_RANDOM_STATE,
    )
    logger.info("Train: %d  Test: %d", len(y_train), len(y_test))

    models     = build_models()
    results    = {}

    # ── Cross-validation ──────────────────────────────────────────────────────
    logger.info("\n── Cross-Validation ─────────────────────────────────────")
    for name, model in models.items():
        cv_acc = cross_validate(name, model, X, y_enc)
        results[name] = {"cv_acc": cv_acc}

    # ── Train and evaluate on held-out test set ────────────────────────────────
    logger.info("\n── Test Set Evaluation ──────────────────────────────────")
    best_name, best_model, best_f1 = None, None, -1.0
    best_cm = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc, f1, cm = evaluate_model(name, model, X_test, y_test, label_names)
        results[name].update({"test_acc": acc, "test_f1": f1})
        if f1 > best_f1:
            best_f1, best_name, best_model, best_cm = f1, name, model, cm

    # ── Confusion matrix for best model ───────────────────────────────────────
    logger.info("\nBest model: %s  (F1=%.4f)", best_name, best_f1)
    cm_path = os.path.join(config.MODELS_DIR, f"confusion_matrix_{best_name}.png")
    plot_confusion_matrix(best_cm, label_names, best_name, save_path=cm_path)

    # ── Save best model ────────────────────────────────────────────────────────
    save_model(best_model, le)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"{'Model':<22}  {'CV Acc':>8}  {'Test Acc':>10}  {'Test F1':>9}")
    for name, res in results.items():
        marker = " ★" if name == best_name else ""
        print(f"{name:<22}  {res['cv_acc']:>8.4f}  {res.get('test_acc', 0):>10.4f}  {res.get('test_f1', 0):>9.4f}{marker}")


if __name__ == "__main__":
    train()
