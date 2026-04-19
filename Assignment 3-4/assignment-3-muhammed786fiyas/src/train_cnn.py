# src/train_cnn.py
# Assisted by Claude

import os
import csv
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.utils.logger import get_logger
from src.utils.config import load_config, load_params

logger = get_logger("train_cnn")


# ─────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Seeds set → {seed}")


# ─────────────────────────────────────────────────────────
# Data Generators
# ─────────────────────────────────────────────────────────

def build_data_generators(config: dict, params: dict):
    """
    Build Keras ImageDataGenerators for train, val, test.
    Train  → on-the-fly augmentation from v1_resized
             generates NEW random augmentations every epoch
             more effective than static augmentation
    Val    → only rescale, no augmentation
    Test   → only rescale, no augmentation
    All paths from config.yaml, all params from params.yaml.
    """
    train_dir  = config["paths"]["train_dir"]
    val_dir    = config["paths"]["val_dir"]
    test_dir   = config["paths"]["test_dir"]
    img_size   = params["prepare"]["img_size"]
    batch_size = params["train_cnn"]["batch_size"]
    seed       = params["train_cnn"]["seed"]

    logger.info(f"Building data generators")
    logger.info(f"  train_dir  → {train_dir}")
    logger.info(f"  val_dir    → {val_dir}")
    logger.info(f"  test_dir   → {test_dir}")
    logger.info(f"  img_size   → {img_size}x{img_size}")
    logger.info(f"  batch_size → {batch_size}")

    # EfficientNet expects pixel values in [0, 255]
    # MobileNetV2 expects pixel values in [-1, 1] via rescale=1/255
    base_model_name = params["train_cnn"]["base_model"]

    if "efficientnet" in base_model_name.lower():
        # EfficientNet has built-in preprocessing
        # Do NOT rescale — pass raw pixel values
        logger.info(f"  Preprocessing → EfficientNet (no rescale needed)")
        rescale_factor = 1.0   # no rescale for EfficientNet
    else:
        # MobileNetV2 needs rescale to [0,1]
        logger.info(f"  Preprocessing → MobileNetV2 (rescale 1/255)")
        rescale_factor = 1.0 / 255

    train_datagen = ImageDataGenerator(
        rescale=rescale_factor,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        fill_mode="nearest"
    )

    val_test_datagen = ImageDataGenerator(rescale=rescale_factor)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = len(train_gen.class_indices)
    logger.info(f"  Number of classes → {num_classes}")
    logger.info(f"  Train samples     → {train_gen.samples}")
    logger.info(f"  Val samples       → {val_gen.samples}")
    logger.info(f"  Test samples      → {test_gen.samples}")

    return train_gen, val_gen, test_gen, num_classes


# ─────────────────────────────────────────────────────────
# Class Weights
# ─────────────────────────────────────────────────────────

def get_class_weights(train_gen, max_weight: float = 5.0) -> dict:
    """
    Compute balanced class weights capped at max_weight.
    Prevents rare classes from dominating training.
    max_weight from params.yaml.
    """
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_gen.classes
    )
    class_weight_dict = dict(zip(classes, weights))

    class_weight_dict = {
        k: min(v, max_weight)
        for k, v in class_weight_dict.items()
    }

    min_w = min(class_weight_dict.values())
    max_w = max(class_weight_dict.values())
    logger.info(
        f"Class weights → "
        f"min={min_w:.3f}, max={max_w:.3f} "
        f"(capped at {max_weight})"
    )

    return class_weight_dict


# ─────────────────────────────────────────────────────────
# Build Model
# ─────────────────────────────────────────────────────────

def build_model(num_classes: int, params: dict):
    """
    Build transfer learning model.
    Supports MobileNetV2 and EfficientNetB0.
    Model selected via params.yaml base_model param.

    Why EfficientNetB0 is better for small datasets:
      - Uses compound scaling (depth + width + resolution)
      - Better feature extraction with fewer parameters
      - Specifically designed for efficiency
      - Consistently outperforms MobileNetV2 on small datasets

    All architecture params from params.yaml.
    """
    base_model_name  = params["train_cnn"]["base_model"]
    learning_rate    = params["train_cnn"]["learning_rate"]
    dropout1         = params["train_cnn"]["dropout1"]
    dropout2         = params["train_cnn"]["dropout2"]

    logger.info(f"Building model")
    logger.info(f"  base_model    → {base_model_name}  ← from params.yaml")
    logger.info(f"  num_classes   → {num_classes}")
    logger.info(f"  learning_rate → {learning_rate}    ← from params.yaml")
    logger.info(f"  dropout1      → {dropout1}         ← from params.yaml")
    logger.info(f"  dropout2      → {dropout2}         ← from params.yaml")

    # ── Select base model from params.yaml ──────────────────
    if base_model_name.lower() == "efficientnetb0":
        base_model = EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
        logger.info(f"  EfficientNetB0 loaded ✅")

    elif base_model_name.lower() == "mobilenetv2":
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
        logger.info(f"  MobileNetV2 loaded ✅")

    else:
        logger.error(f"Unknown base_model: {base_model_name}")
        raise ValueError(
            f"base_model must be 'efficientnetb0' or 'mobilenetv2', "
            f"got: {base_model_name}"
        )

    # Freeze entire base for phase 1
    base_model.trainable = False
    logger.info(f"  Base model layers → {len(base_model.layers)}")
    logger.info(f"  Base model fully frozen for phase 1")

    # Custom classification head
    x      = base_model.output
    x      = GlobalAveragePooling2D()(x)
    x      = BatchNormalization()(x)
    x      = Dense(256, activation="relu")(x)
    x      = Dropout(dropout1)(x)        # ← from params.yaml
    x      = Dense(128, activation="relu")(x)
    x      = Dropout(dropout2)(x)        # ← from params.yaml
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    total     = model.count_params()
    logger.info(f"  Total params     → {total:,}")
    logger.info(f"  Trainable params → {trainable:,}")

    return model, base_model


# ─────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────

def build_callbacks(models_dir: Path, patience: int = 7) -> list:
    """
    ONE best model path shared across both phases.
    ModelCheckpoint only saves when val_accuracy improves globally.
    """
    best_model_path = str(models_dir / "cnn_best.h5")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    logger.info(
        f"Callbacks → "
        f"EarlyStopping(patience={patience}), "
        f"ReduceLROnPlateau(patience=4, factor=0.5), "
        f"ModelCheckpoint → {best_model_path}"
    )

    return callbacks


# ─────────────────────────────────────────────────────────
# Phase 1 — Train Head Only (Frozen Base)
# ─────────────────────────────────────────────────────────

def phase1_train(
    model, base_model,
    train_gen, val_gen,
    params: dict,
    models_dir: Path,
    class_weight_dict: dict
):
    """
    Phase 1: Train only custom head with fully frozen base.
    Head learns to use pretrained ImageNet features.
    phase1_epochs from params.yaml.
    """
    phase1_epochs = params["train_cnn"]["phase1_epochs"]
    trainable     = sum(tf.size(w).numpy() for w in model.trainable_weights)

    logger.info(f"Phase 1 — frozen base")
    logger.info(f"  phase1_epochs    → {phase1_epochs}  ← from params.yaml")
    logger.info(f"  Trainable params → {trainable:,}")

    callbacks = build_callbacks(models_dir, patience=7)

    history = model.fit(
        train_gen,
        epochs=phase1_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    best_val_acc = max(history.history["val_accuracy"])
    epochs_run   = len(history.history["accuracy"])
    logger.info(f"Phase 1 complete ✓")
    logger.info(f"  Epochs run        → {epochs_run}/{phase1_epochs}")
    logger.info(f"  Best val_accuracy → {best_val_acc:.4f}")

    return history


# ─────────────────────────────────────────────────────────
# Phase 2 — Gentle Fine-Tuning (Optional)
# ─────────────────────────────────────────────────────────

def phase2_finetune(
    model, base_model,
    train_gen, val_gen,
    params: dict,
    models_dir: Path,
    class_weight_dict: dict
):
    """
    Phase 2: Optionally unfreeze top N layers and fine-tune.

    BUG FIX: unfreeze_layers=0 skips phase 2 entirely.
    Python: list[-0:] = entire list (not empty!)
    Fix: explicit check for unfreeze_layers > 0.

    BatchNorm layers always kept frozen to prevent
    distribution shift during fine-tuning.

    Returns None if phase 2 is skipped.
    """
    unfreeze_layers = params["train_cnn"]["unfreeze_layers"]
    finetune_lr     = params["train_cnn"]["finetune_lr"]
    epochs          = params["train_cnn"]["epochs"]

    # ── Skip phase 2 if unfreeze_layers=0 ───────────────────
    if unfreeze_layers == 0:
        logger.info("Phase 2 SKIPPED — unfreeze_layers=0 in params.yaml")
        return None

    logger.info(f"Phase 2 — gentle fine-tuning")
    logger.info(f"  unfreeze_layers → {unfreeze_layers}  ← from params.yaml")
    logger.info(f"  finetune_lr     → {finetune_lr}      ← from params.yaml")
    logger.info(f"  max epochs      → {epochs}           ← from params.yaml")

    # Unfreeze top N layers explicitly
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    # Keep ALL BatchNorm frozen — prevents distribution shift
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    logger.info(f"  Trainable params now → {trainable:,}")

    model.compile(
        optimizer=Adam(learning_rate=finetune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = build_callbacks(models_dir, patience=8)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    best_val_acc = max(history.history["val_accuracy"])
    epochs_run   = len(history.history["accuracy"])
    logger.info(f"Phase 2 complete ✓")
    logger.info(f"  Epochs run        → {epochs_run}/{epochs}")
    logger.info(f"  Best val_accuracy → {best_val_acc:.4f}")

    return history


# ─────────────────────────────────────────────────────────
# Evaluate on Validation
# ─────────────────────────────────────────────────────────

def evaluate_on_val(model, val_gen) -> tuple:
    """Evaluate model on validation set."""
    logger.info("Evaluating CNN on validation set...")

    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = val_gen.classes

    accuracy = np.mean(y_pred == y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    idx_to_class = {v: k for k, v in val_gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0
    )

    logger.info(f"  Validation Accuracy → {accuracy:.4f}")
    logger.info(f"  Validation Macro F1 → {macro_f1:.4f}")
    logger.info(f"  Per-class report:\n{report}")

    return accuracy, macro_f1


# ─────────────────────────────────────────────────────────
# Save Training History
# ─────────────────────────────────────────────────────────

def save_history(history1, history2, metrics_dir: Path):
    """
    Combine phase 1 and phase 2 histories.
    If phase 2 skipped (history2=None) saves phase 1 only.
    Saves as CSV for dvc plots.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if history2 is None:
        combined   = history1.history
        total_ep   = len(combined["accuracy"])
        logger.info("Saving phase 1 history only (phase 2 skipped)")
    else:
        combined = {}
        for key in history1.history:
            combined[key] = (
                history1.history[key] +
                history2.history.get(key, [])
            )
        total_ep = len(combined["accuracy"])

    csv_path = metrics_dir / "cnn_training_history.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "accuracy", "val_accuracy",
                "loss", "val_loss"
            ]
        )
        writer.writeheader()
        for i in range(total_ep):
            writer.writerow({
                "epoch"        : i + 1,
                "accuracy"     : round(combined["accuracy"][i], 4),
                "val_accuracy" : round(combined["val_accuracy"][i], 4),
                "loss"         : round(combined["loss"][i], 4),
                "val_loss"     : round(combined["val_loss"][i], 4),
            })

    logger.info(f"Training history saved → {csv_path} ({total_ep} epochs)")
    return csv_path


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def train_cnn():
    logger.info("=" * 55)
    logger.info("TRAIN CNN STAGE STARTED")
    logger.info("=" * 55)

    config = load_config()
    params = load_params()

    models_dir       = Path(config["paths"]["models_dir"])
    model_path       = Path(config["paths"]["cnn_model"])
    metrics_dir      = Path(config["paths"]["metrics_dir"])
    cnn_params       = params["train_cnn"]
    seed             = cnn_params["seed"]
    unfreeze_layers  = cnn_params["unfreeze_layers"]
    max_class_weight = cnn_params.get("max_class_weight", 5.0)

    logger.info(f"Paths from config.yaml")
    logger.info(f"  models_dir  → {models_dir}")
    logger.info(f"  model_path  → {model_path}")
    logger.info(f"  metrics_dir → {metrics_dir}")
    logger.info(f"CNN params from params.yaml")
    logger.info(f"  base_model       → {cnn_params['base_model']}")
    logger.info(f"  phase1_epochs    → {cnn_params['phase1_epochs']}")
    logger.info(f"  epochs           → {cnn_params['epochs']}")
    logger.info(f"  batch_size       → {cnn_params['batch_size']}")
    logger.info(f"  learning_rate    → {cnn_params['learning_rate']}")
    logger.info(f"  unfreeze_layers  → {unfreeze_layers}")
    logger.info(f"  finetune_lr      → {cnn_params['finetune_lr']}")
    logger.info(f"  dropout1         → {cnn_params['dropout1']}")
    logger.info(f"  dropout2         → {cnn_params['dropout2']}")
    logger.info(f"  max_class_weight → {max_class_weight}")
    logger.info(f"  seed             → {seed}")
    logger.info(
        f"  Phase 2 → "
        f"{'ENABLED (unfreeze top ' + str(unfreeze_layers) + ' layers)' if unfreeze_layers > 0 else 'DISABLED'}"
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Seeds ────────────────────────────────────────
    logger.info("Step 1: Setting seeds")
    set_seed(seed)

    # ── Step 2: Data generators ──────────────────────────────
    logger.info("Step 2: Building data generators")
    train_gen, val_gen, test_gen, num_classes = build_data_generators(
        config, params
    )

    # ── Step 3: Class weights ────────────────────────────────
    logger.info("Step 3: Computing class weights")
    class_weight_dict = get_class_weights(train_gen, max_class_weight)

    # ── Step 4: Build model ──────────────────────────────────
    logger.info("Step 4: Building model")
    model, base_model = build_model(num_classes, params)

    # ── Step 5: Phase 1 ──────────────────────────────────────
    logger.info("Step 5: Phase 1 — training head only")
    history1 = phase1_train(
        model, base_model,
        train_gen, val_gen,
        params, models_dir,
        class_weight_dict
    )

    # ── Step 6: Phase 2 (optional) ───────────────────────────
    logger.info("Step 6: Phase 2 — fine-tuning")
    history2 = phase2_finetune(
        model, base_model,
        train_gen, val_gen,
        params, models_dir,
        class_weight_dict
    )

    # ── Step 7: Evaluate ─────────────────────────────────────
    logger.info("Step 7: Evaluating on validation set")
    val_accuracy, val_macro_f1 = evaluate_on_val(model, val_gen)

    # ── Step 8: Save history ─────────────────────────────────
    logger.info("Step 8: Saving training history")
    save_history(history1, history2, metrics_dir)

    # ── Step 9: Save model ───────────────────────────────────
    logger.info("Step 9: Saving final model")
    model.save(str(model_path))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Model saved → {model_path} ({size_mb:.2f} MB)")

    # ── Summary ──────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("TRAIN CNN SUMMARY")
    logger.info(f"  Base model        → {cnn_params['base_model']}")
    logger.info(f"  Phase 1 best val_accuracy → {max(history1.history['val_accuracy']):.4f}")
    if history2 is not None:
        logger.info(f"  Phase 2 best val_accuracy → {max(history2.history['val_accuracy']):.4f}")
    else:
        logger.info(f"  Phase 2                   → SKIPPED")
    logger.info(f"  Final Validation Accuracy → {val_accuracy:.4f}")
    logger.info(f"  Final Validation Macro F1 → {val_macro_f1:.4f}")
    logger.info(f"  Model saved to            → {model_path}")
    logger.info("=" * 55)
    logger.info("TRAIN CNN STAGE COMPLETE ✓")
    logger.info("=" * 55)


if __name__ == "__main__":
    train_cnn()