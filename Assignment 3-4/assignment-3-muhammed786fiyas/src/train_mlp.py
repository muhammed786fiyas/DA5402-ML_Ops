# src/train_mlp.py
# Assisted by Claude

import pickle
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.utils.logger import get_logger
from src.utils.config import load_config, load_params

logger = get_logger("train_mlp")


# ─────────────────────────────────────────────────────────
# Load HOG Features
# ─────────────────────────────────────────────────────────

def load_features(config: dict):
    """
    Load HOG features and labels for train and val splits.
    All paths loaded from config.yaml — no hardcoded paths.
    """
    features_train = Path(config["paths"]["features_train"])
    features_val   = Path(config["paths"]["features_val"])
    labels_train   = Path(config["paths"]["labels_train"])
    labels_val     = Path(config["paths"]["labels_val"])

    logger.info(f"Loading features from config.yaml paths")
    logger.info(f"  features_train → {features_train}")
    logger.info(f"  features_val   → {features_val}")
    logger.info(f"  labels_train   → {labels_train}")
    logger.info(f"  labels_val     → {labels_val}")

    X_train = np.load(features_train)
    y_train = np.load(labels_train)
    X_val   = np.load(features_val)
    y_val   = np.load(labels_val)

    logger.info(f"  X_train shape  : {X_train.shape}")
    logger.info(f"  y_train shape  : {y_train.shape}")
    logger.info(f"  X_val shape    : {X_val.shape}")
    logger.info(f"  y_val shape    : {y_val.shape}")

    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────
# Build MLP Pipeline
# ─────────────────────────────────────────────────────────

def build_pipeline(params: dict) -> Pipeline:
    """
    Build sklearn Pipeline with:
      1. StandardScaler  — normalize HOG features
      2. PCA             — reduce 26244 dims to pca_components
      3. MLPClassifier   — Multi Layer Perceptron

    Why PCA before MLP?
    26244 features is too many for MLP — leads to slow training
    and overfitting. PCA reduces to most important components.

    Args:
        params: train_mlp section from params.yaml

    Returns:
        sklearn Pipeline
    """
    hidden_layers  = tuple(params["hidden_layers"])
    max_iter       = params["max_iter"]
    seed           = params["seed"]
    pca_components = params["pca_components"]
    learning_rate  = params["learning_rate_init"]

    logger.info(
        f"Building MLP pipeline → "
        f"hidden_layers={hidden_layers}, "
        f"max_iter={max_iter}, "
        f"pca_components={pca_components}, "
        f"learning_rate={learning_rate}, "
        f"seed={seed}"
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(
            n_components=pca_components,
            random_state=seed
        )),
        ("mlp",    MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=seed,
            learning_rate_init=learning_rate,
            early_stopping=True,      # stops if val score stops improving
            validation_fraction=0.1,  # uses 10% of train for early stopping
            n_iter_no_change=10,      # patience
            verbose=False
        ))
    ])

    return pipeline


# ─────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────

def train(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit the MLP pipeline on training data.
    PCA and Scaler fit on train only — no data leakage.
    """
    logger.info(f"Fitting MLP on {X_train.shape[0]} training samples...")
    logger.info("This may take a few minutes...")

    pipeline.fit(X_train, y_train)

    mlp = pipeline.named_steps["mlp"]
    logger.info(f"MLP training complete ✓")
    logger.info(f"  Iterations run  : {mlp.n_iter_}")
    logger.info(f"  Final loss      : {mlp.loss_:.4f}")
    logger.info(f"  Best val score  : {mlp.best_validation_score_:.4f}")

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
    Evaluate trained MLP on validation set.
    Reports accuracy, macro F1, and per-class report.
    """
    logger.info("Evaluating MLP on validation set...")

    y_pred = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    # Reverse label map: {0: "Brindavan", 1: "Gir", ...}
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
    """Save trained MLP pipeline as .pkl file."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model saved → {model_path} ({size_mb:.2f} MB)")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def train_mlp():
    logger.info("=" * 55)
    logger.info("TRAIN MLP STAGE STARTED")
    logger.info("=" * 55)

    # ── Load config and params ───────────────────────────────
    config = load_config()
    params = load_params()

    # Paths from config.yaml
    features_dir = Path(config["paths"]["features_dir"])
    model_path   = Path(config["paths"]["mlp_model"])

    # Params from params.yaml
    mlp_params = params["train_mlp"]

    logger.info(f"Paths loaded from config.yaml")
    logger.info(f"  features_dir → {features_dir}")
    logger.info(f"  model_path   → {model_path}")
    logger.info(
        f"MLP params loaded from params.yaml → "
        f"hidden_layers={mlp_params['hidden_layers']}, "
        f"max_iter={mlp_params['max_iter']}, "
        f"pca_components={mlp_params['pca_components']}, "
        f"learning_rate={mlp_params['learning_rate_init']}, "
        f"seed={mlp_params['seed']}"
    )

    # Validate features exist
    if not features_dir.exists():
        logger.error(
            f"Features not found at {features_dir}. "
            f"Run transform.py first."
        )
        raise FileNotFoundError(f"Features not found: {features_dir}")

    # ── Step 1: Load features ────────────────────────────────
    logger.info("Step 1: Loading HOG features")
    X_train, y_train, X_val, y_val = load_features(config)

    # Load label map
    label_map_path = features_dir / "label_map.npy"
    label_map = np.load(
        label_map_path,
        allow_pickle=True
    ).item()
    logger.info(f"Label map loaded from {label_map_path} → {len(label_map)} classes")

    # ── Step 2: Build pipeline ───────────────────────────────
    logger.info("Step 2: Building MLP pipeline")
    pipeline = build_pipeline(mlp_params)

    # ── Step 3: Train ────────────────────────────────────────
    logger.info("Step 3: Training MLP")
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
    logger.info("TRAIN MLP SUMMARY")
    logger.info(f"  Validation Accuracy : {val_accuracy:.4f}")
    logger.info(f"  Validation Macro F1 : {val_macro_f1:.4f}")
    logger.info(f"  Model saved to      : {model_path}")
    logger.info("=" * 55)
    logger.info("TRAIN MLP STAGE COMPLETE ✓")
    logger.info("=" * 55)


if __name__ == "__main__":
    train_mlp()