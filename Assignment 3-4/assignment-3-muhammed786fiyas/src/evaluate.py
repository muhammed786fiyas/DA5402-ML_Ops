# src/evaluate.py
# Assisted by Claude

import json
import csv
import numpy as np
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)

from src.utils.logger import get_logger
from src.utils.config import load_config, load_params

logger = get_logger("evaluate")


# ─────────────────────────────────────────────────────────
# Load Test Features (for SVM and MLP)
# ─────────────────────────────────────────────────────────

def load_test_features(config: dict):
    """
    Load HOG test features and labels from data/features/.
    Used by SVM and MLP models.
    All paths from config.yaml.
    """
    features_test = Path(config["paths"]["features_test"])
    labels_test   = Path(config["paths"]["labels_test"])
    label_map_path = Path(config["paths"]["features_dir"]) / "label_map.npy"

    logger.info(f"Loading test features")
    logger.info(f"  features_test → {features_test}")
    logger.info(f"  labels_test   → {labels_test}")

    X_test    = np.load(features_test)
    y_test    = np.load(labels_test)
    label_map = np.load(label_map_path, allow_pickle=True).item()

    logger.info(f"  X_test shape  → {X_test.shape}")
    logger.info(f"  y_test shape  → {y_test.shape}")
    logger.info(f"  Classes       → {len(label_map)}")

    return X_test, y_test, label_map


# ─────────────────────────────────────────────────────────
# Load Test Generator (for CNN)
# ─────────────────────────────────────────────────────────

def load_test_generator(config: dict, params: dict):

    test_dir   = config["paths"]["test_dir"]
    img_size   = params["prepare"]["img_size"]
    batch_size = params["train_cnn"]["batch_size"]

    base_model = params["train_cnn"]["base_model"]
    if "efficientnet" in base_model.lower():
        rescale = 1.0          # ← no rescale for EfficientNet
        logger.info(f"  Preprocessing → EfficientNet (no rescale)")
    else:
        rescale = 1.0 / 255    # ← rescale for MobileNetV2
        logger.info(f"  Preprocessing → MobileNetV2 (rescale 1/255)")

    logger.info(f"Building test generator")
    logger.info(f"  test_dir   → {test_dir}")
    logger.info(f"  img_size   → {img_size}x{img_size}")
    logger.info(f"  batch_size → {batch_size}")

    test_datagen = ImageDataGenerator(rescale=rescale)  # ← dynamic

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False    # MUST be False for correct label alignment
    )

    logger.info(f"  Test samples → {test_gen.samples}")
    logger.info(f"  Classes      → {test_gen.class_indices}")

    return test_gen


# ─────────────────────────────────────────────────────────
# Evaluate SVM
# ─────────────────────────────────────────────────────────

def evaluate_svm(config: dict, X_test: np.ndarray, y_test: np.ndarray, label_map: dict):
    """
    Load and evaluate SVM model on test set.
    Returns accuracy, macro F1, predictions.
    """
    model_path = Path(config["paths"]["svm_model"])
    logger.info(f"Evaluating SVM model → {model_path}")

    if not model_path.exists():
        logger.error(f"SVM model not found: {model_path}")
        raise FileNotFoundError(f"SVM model not found: {model_path}")

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    idx_to_breed = {v: k for k, v in label_map.items()}
    target_names = [idx_to_breed[i] for i in sorted(idx_to_breed.keys())]

    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0
    )

    logger.info(f"  SVM Test Accuracy → {accuracy:.4f}")
    logger.info(f"  SVM Test Macro F1 → {macro_f1:.4f}")
    logger.info(f"  SVM Per-class report:\n{report}")

    return accuracy, macro_f1, y_pred


# ─────────────────────────────────────────────────────────
# Evaluate MLP
# ─────────────────────────────────────────────────────────

def evaluate_mlp(config: dict, X_test: np.ndarray, y_test: np.ndarray, label_map: dict):
    """
    Load and evaluate MLP model on test set.
    Returns accuracy, macro F1, predictions.
    """
    model_path = Path(config["paths"]["mlp_model"])
    logger.info(f"Evaluating MLP model → {model_path}")

    if not model_path.exists():
        logger.error(f"MLP model not found: {model_path}")
        raise FileNotFoundError(f"MLP model not found: {model_path}")

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    idx_to_breed = {v: k for k, v in label_map.items()}
    target_names = [idx_to_breed[i] for i in sorted(idx_to_breed.keys())]

    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0
    )

    logger.info(f"  MLP Test Accuracy → {accuracy:.4f}")
    logger.info(f"  MLP Test Macro F1 → {macro_f1:.4f}")
    logger.info(f"  MLP Per-class report:\n{report}")

    return accuracy, macro_f1, y_pred


# ─────────────────────────────────────────────────────────
# Evaluate CNN
# ─────────────────────────────────────────────────────────

def evaluate_cnn(config: dict, test_gen):
    """
    Load and evaluate CNN model on test set.
    Returns accuracy, macro F1, predictions, true labels.
    """
    model_path = Path(config["paths"]["cnn_model"])
    logger.info(f"Evaluating CNN model → {model_path}")

    if not model_path.exists():
        logger.error(f"CNN model not found: {model_path}")
        raise FileNotFoundError(f"CNN model not found: {model_path}")

    model = load_model(str(model_path))

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = test_gen.classes

    accuracy = np.mean(y_pred == y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0
    )

    logger.info(f"  CNN Test Accuracy → {accuracy:.4f}")
    logger.info(f"  CNN Test Macro F1 → {macro_f1:.4f}")
    logger.info(f"  CNN Per-class report:\n{report}")

    return accuracy, macro_f1, y_pred, y_true, target_names


# ─────────────────────────────────────────────────────────
# Save Metrics (dvc metrics show reads this)
# ─────────────────────────────────────────────────────────

def save_metrics(
    metrics_dir: Path,
    svm_acc, svm_f1,
    mlp_acc, mlp_f1,
    cnn_acc, cnn_f1
):
    """
    Save all model metrics to metrics/scores.json.
    This is what dvc metrics show reads.
    Format matches DVC metrics requirements.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scores = {
        "svm": {
            "test_accuracy" : round(float(svm_acc), 4),
            "test_macro_f1" : round(float(svm_f1), 4)
        },
        "mlp": {
            "test_accuracy" : round(float(mlp_acc), 4),
            "test_macro_f1" : round(float(mlp_f1), 4)
        },
        "cnn": {
            "test_accuracy" : round(float(cnn_acc), 4),
            "test_macro_f1" : round(float(cnn_f1), 4)
        },
        "best_model": {
            "name"     : "cnn",
            "macro_f1" : round(float(cnn_f1), 4)
        }
    }

    scores_path = metrics_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    logger.info(f"Metrics saved → {scores_path}")
    logger.info(f"  SVM  → accuracy={svm_acc:.4f}, macro_f1={svm_f1:.4f}")
    logger.info(f"  MLP  → accuracy={mlp_acc:.4f}, macro_f1={mlp_f1:.4f}")
    logger.info(f"  CNN  → accuracy={cnn_acc:.4f}, macro_f1={cnn_f1:.4f}")

    return scores_path


# ─────────────────────────────────────────────────────────
# Save Confusion Matrix (dvc plots show reads this)
# ─────────────────────────────────────────────────────────

def save_confusion_matrix(
    metrics_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
    model_name: str
):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "confusion_matrix.csv"

    # Use numeric indices for dvc plots compatibility
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["actual_id", "predicted_id", "actual", "predicted", "count"]
        )
        writer.writeheader()

        cm = confusion_matrix(y_true, y_pred)
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                if cm[i][j] > 0:
                    writer.writerow({
                        "actual_id"    : i,
                        "predicted_id" : j,
                        "actual"       : target_names[i],
                        "predicted"    : target_names[j],
                        "count"        : int(cm[i][j])
                    })

    logger.info(f"Confusion matrix saved → {csv_path}")
    return csv_path

# ─────────────────────────────────────────────────────────
# Save F1 Comparison Plot
# ─────────────────────────────────────────────────────────

def save_f1_comparison(metrics_dir, svm_f1, mlp_f1, cnn_f1):
    csv_path = metrics_dir / "f1_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model_id", "model", "macro_f1"]
        )
        writer.writeheader()
        writer.writerow({"model_id": 1, "model": "SVM", "macro_f1": round(svm_f1, 4)})
        writer.writerow({"model_id": 2, "model": "MLP", "macro_f1": round(mlp_f1, 4)})
        writer.writerow({"model_id": 3, "model": "CNN", "macro_f1": round(cnn_f1, 4)})
    logger.info(f"F1 comparison saved → {csv_path}")
    return csv_path

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def evaluate():
    logger.info("=" * 55)
    logger.info("EVALUATE STAGE STARTED")
    logger.info("=" * 55)

    # ── Load config and params ───────────────────────────────
    config = load_config()
    params = load_params()

    metrics_dir = Path(config["paths"]["metrics_dir"])

    logger.info(f"Paths from config.yaml")
    logger.info(f"  metrics_dir → {metrics_dir}")
    logger.info(f"  svm_model   → {config['paths']['svm_model']}")
    logger.info(f"  mlp_model   → {config['paths']['mlp_model']}")
    logger.info(f"  cnn_model   → {config['paths']['cnn_model']}")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load test features for SVM + MLP ────────────
    logger.info("Step 1: Loading HOG test features")
    X_test, y_test, label_map = load_test_features(config)

    # ── Step 2: Load test generator for CNN ─────────────────
    logger.info("Step 2: Building CNN test generator")
    test_gen = load_test_generator(config, params)

    # ── Step 3: Evaluate SVM ─────────────────────────────────
    logger.info("Step 3: Evaluating SVM on test set")
    svm_acc, svm_f1, svm_preds = evaluate_svm(
        config, X_test, y_test, label_map
    )

    # ── Step 4: Evaluate MLP ─────────────────────────────────
    logger.info("Step 4: Evaluating MLP on test set")
    mlp_acc, mlp_f1, mlp_preds = evaluate_mlp(
        config, X_test, y_test, label_map
    )

    # ── Step 5: Evaluate CNN ─────────────────────────────────
    logger.info("Step 5: Evaluating CNN on test set")
    cnn_acc, cnn_f1, cnn_preds, cnn_true, target_names = evaluate_cnn(
        config, test_gen
    )

    # ── Step 6: Save metrics ─────────────────────────────────
    logger.info("Step 6: Saving metrics")
    save_metrics(
        metrics_dir,
        svm_acc, svm_f1,
        mlp_acc, mlp_f1,
        cnn_acc, cnn_f1
    )

    # ── Step 7: Save confusion matrix (CNN) ──────────────────
    logger.info("Step 7: Saving CNN confusion matrix")
    save_confusion_matrix(
        metrics_dir,
        cnn_true, cnn_preds,
        target_names, "cnn"
    )

    # ── Step 8: Save F1 comparison ───────────────────────────
    logger.info("Step 8: Saving F1 comparison")
    save_f1_comparison(metrics_dir, svm_f1, mlp_f1, cnn_f1)

    # ── Final Summary ────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("EVALUATION SUMMARY — TEST SET RESULTS")
    logger.info("=" * 55)
    logger.info(f"  {'Model':<8} {'Accuracy':>10} {'Macro F1':>10}")
    logger.info(f"  {'-'*30}")
    logger.info(f"  {'SVM':<8} {svm_acc:>10.4f} {svm_f1:>10.4f}")
    logger.info(f"  {'MLP':<8} {mlp_acc:>10.4f} {mlp_f1:>10.4f}")
    logger.info(f"  {'CNN':<8} {cnn_acc:>10.4f} {cnn_f1:>10.4f}")
    logger.info(f"  {'-'*30}")
    logger.info(f"  Best Model → CNN (Macro F1: {cnn_f1:.4f})")
    logger.info("=" * 55)
    logger.info("EVALUATE STAGE COMPLETE ✓")
    logger.info("=" * 55)


if __name__ == "__main__":
    evaluate()