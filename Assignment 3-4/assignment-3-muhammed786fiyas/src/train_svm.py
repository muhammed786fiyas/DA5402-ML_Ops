# src/train_svm.py
# Assisted by Claude

import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger
from src.utils.config import load_config, load_params

logger = get_logger("train_svm")


# ─────────────────────────────────────────────────────────
# Load HOG Features
# ─────────────────────────────────────────────────────────

def load_features(features_dir: Path):
    """
    Load HOG features and labels for train, val splits.
    These were created by transform.py.
    """
    logger.info(f"Loading HOG features from: {features_dir}")

    X_train = np.load(features_dir / "hog_train.npy")
    y_train = np.load(features_dir / "labels_train.npy")
    X_val   = np.load(features_dir / "hog_val.npy")
    y_val   = np.load(features_dir / "labels_val.npy")

    logger.info(f"  X_train shape : {X_train.shape}")
    logger.info(f"  y_train shape : {y_train.shape}")
    logger.info(f"  X_val shape   : {X_val.shape}")
    logger.info(f"  y_val shape   : {y_val.shape}")

    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────
# Build SVM Pipeline
# ─────────────────────────────────────────────────────────

def build_pipeline(params: dict) -> Pipeline:
    """
    Build an sklearn Pipeline with:
      1. StandardScaler  — normalizes HOG features (zero mean, unit variance)
      2. SVC             — Support Vector Classifier

    Why StandardScaler before SVM?
    SVM is sensitive to feature scale. HOG features can have very different
    ranges — scaling them ensures SVM works correctly.

    Args:
        params: train_svm section from params.yaml

    Returns:
        sklearn Pipeline
    """
    C      = params["C"]
    kernel = params["kernel"]
    seed   = params["seed"]

    logger.info(f"Building SVM pipeline → C={C}, kernel={kernel}, seed={seed}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            C=C,
            kernel=kernel,
            random_state=seed,
            probability=True,    # needed for predict_proba
            verbose=False
        ))
    ])

    return pipeline


# ─────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────

def train(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit the SVM pipeline on training data.
    StandardScaler fits on train data only — prevents data leakage.
    """
    logger.info(f"Fitting SVM on {X_train.shape[0]} training samples...")
    logger.info("This may take a few minutes for large datasets...")

    pipeline.fit(X_train, y_train)

    logger.info("SVM fitting complete ✓")
    return pipeline


# ─────────────────────────────────────────────────────────
# Evaluate on Validation Set
# ─────────────────────────────────────────────────────────

def evaluate_on_val(
    pipeline: Pipeline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_map: dict
):
    """
    Evaluate trained SVM on validation set.
    Reports accuracy, macro F1, and per-class report.
    """
    logger.info("Evaluating SVM on validation set...")

    y_pred = pipeline.predict(X_val)

    accuracy  = accuracy_score(y_val, y_pred)
    macro_f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)

    # Reverse label map for readable report: {0: "Brindavan", 1: "Gir", ...}
    idx_to_breed = {v: k for k, v in label_map.items()}
    target_names = [idx_to_breed[i] for i in sorted(idx_to_breed.keys())]

    report = classification_report(
        y_val, y_pred,
        target_names=target_names,
        zero_division=0
    )

    logger.info(f"  Validation Accuracy : {accuracy:.4f}")
    logger.info(f"  Validation Macro F1 : {macro_f1:.4f}")
    logger.info(f"  Per-class report:\n{report}")

    return accuracy, macro_f1


# ─────────────────────────────────────────────────────────
# Save Model
# ─────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, model_path: Path):
    """Save trained SVM pipeline as .pkl file."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model saved → {model_path} ({size_mb:.2f} MB)")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def train_svm():
    logger.info("=" * 55)
    logger.info("TRAIN SVM STAGE STARTED")
    logger.info("=" * 55)

    # ── Load config and params ───────────────────────────────
    config = load_config()
    params = load_params()

    # Paths from config.yaml
    features_dir = Path(config["paths"]["features_dir"])
    model_path   = Path(config["paths"]["svm_model"])

    # Params from params.yaml
    svm_params = params["train_svm"]

    logger.info(f"Paths loaded from config.yaml")
    logger.info(f"  features_dir → {features_dir}")
    logger.info(f"  model_path   → {model_path}")
    logger.info(
        f"SVM params → C={svm_params['C']}, "
        f"kernel={svm_params['kernel']}, "
        f"seed={svm_params['seed']}"
    )

    # Validate features exist
    if not features_dir.exists():
        logger.error(f"Features not found at {features_dir}. Run transform.py first.")
        raise FileNotFoundError(f"Features not found: {features_dir}")

    # ── Step 1: Load features ────────────────────────────────
    logger.info("Step 1: Loading HOG features")
    X_train, y_train, X_val, y_val = load_features(features_dir)

    # Load label map for readable evaluation report
    label_map = np.load(
        features_dir / "label_map.npy",
        allow_pickle=True
    ).item()
    logger.info(f"Label map loaded → {len(label_map)} classes")

    # ── Step 2: Build pipeline ───────────────────────────────
    logger.info("Step 2: Building SVM pipeline")
    pipeline = build_pipeline(svm_params)

    # ── Step 3: Train ────────────────────────────────────────
    logger.info("Step 3: Training SVM")
    pipeline = train(pipeline, X_train, y_train)

    # ── Step 4: Evaluate on validation ──────────────────────
    logger.info("Step 4: Evaluating on validation set")
    val_accuracy, val_macro_f1 = evaluate_on_val(
        pipeline, X_val, y_val, label_map
    )

    # ── Step 5: Save model ───────────────────────────────────
    logger.info("Step 5: Saving model")
    save_model(pipeline, model_path)

    # ── Summary ──────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("TRAIN SVM SUMMARY")
    logger.info(f"  Validation Accuracy : {val_accuracy:.4f}")
    logger.info(f"  Validation Macro F1 : {val_macro_f1:.4f}")
    logger.info(f"  Model saved to      : {model_path}")
    logger.info("=" * 55)
    logger.info("TRAIN SVM STAGE COMPLETE ✓")
    logger.info("=" * 55)


if __name__ == "__main__":
    train_svm()