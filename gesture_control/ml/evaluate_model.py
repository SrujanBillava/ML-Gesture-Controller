"""
ml/evaluate_model.py
Evaluate the saved model's F1 score without retraining.

Run:
    python ml/evaluate_model.py
"""

import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from ml.train_model import load_dataset


def main():
    # ── Load saved model ──────────────────────────────────────────────────
    if not os.path.exists(config.MODEL_PATH):
        print("[ERROR] No saved model found. Run 'python ml/train_model.py' first.")
        return

    model = joblib.load(config.MODEL_PATH)
    le    = joblib.load(config.LABEL_PATH)
    print("[OK] Model loaded: %s" % config.MODEL_PATH)
    print(f"   Classes: {le.classes_.tolist()}\n")

    # ── Load dataset ──────────────────────────────────────────────────────
    X, y = load_dataset()
    y_enc = le.transform(y)
    label_names = le.classes_.tolist()

    # ── Split (same seed as training for fair comparison) ──────────────────
    _, X_test, _, y_test = train_test_split(
        X, y_enc,
        test_size=config.ML_TEST_SIZE,
        stratify=y_enc,
        random_state=config.ML_RANDOM_STATE,
    )

    # ── Predict ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────
    acc       = accuracy_score(y_test, y_pred)
    f1_w      = f1_score(y_test, y_pred, average="weighted")
    f1_macro  = f1_score(y_test, y_pred, average="macro")
    prec_w    = precision_score(y_test, y_pred, average="weighted")
    recall_w  = recall_score(y_test, y_pred, average="weighted")

    print("=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision (weighted): {prec_w:.4f}")
    print(f"  Recall    (weighted): {recall_w:.4f}")
    print(f"  F1 Score  (weighted): {f1_w:.4f}")
    print(f"  F1 Score  (macro)  : {f1_macro:.4f}")
    print("=" * 60)

    # ── Per-class report ──────────────────────────────────────────────────
    print("\nPer-class breakdown:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    save_path = os.path.join(config.MODELS_DIR, "evaluation_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print("\n[SAVED] Confusion matrix: %s" % save_path)
    plt.show()


if __name__ == "__main__":
    main()
