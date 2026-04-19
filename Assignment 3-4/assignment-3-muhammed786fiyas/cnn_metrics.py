# cnn_metrics.py
# Compute comprehensive metrics for best CNN model
# Run from project root: python cnn_metrics.py

import numpy as np
import yaml
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────────────────
# Load config and params
# ─────────────────────────────────────────────────────────

with open("config.yaml") as f:
    config = yaml.safe_load(f)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

IMG_SIZE   = params["prepare"]["img_size"]
BATCH_SIZE = params["train_cnn"]["batch_size"]
BASE_MODEL = params["train_cnn"]["base_model"]

TRAIN_DIR  = config["paths"]["train_dir"]
VAL_DIR    = config["paths"]["val_dir"]
TEST_DIR   = config["paths"]["test_dir"]
MODEL_PATH = config["paths"]["cnn_model"]

# EfficientNet → no rescale
RESCALE = 1.0 if "efficientnet" in BASE_MODEL.lower() else 1.0 / 255
print(f"Base model     : {BASE_MODEL}")
print(f"Rescale factor : {RESCALE}")
print(f"Model path     : {MODEL_PATH}")
print()


# ─────────────────────────────────────────────────────────
# Build data generators
# ─────────────────────────────────────────────────────────

def make_generator(directory, shuffle=False):
    datagen = ImageDataGenerator(rescale=RESCALE)
    return datagen.flow_from_directory(
        directory,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=shuffle
    )

print("Loading data generators...")
train_gen = make_generator(TRAIN_DIR, shuffle=False)
val_gen   = make_generator(VAL_DIR,   shuffle=False)
test_gen  = make_generator(TEST_DIR,  shuffle=False)

NUM_CLASSES = len(train_gen.class_indices)
print(f"Number of classes: {NUM_CLASSES}")
print(f"Train samples    : {train_gen.samples}")
print(f"Val samples      : {val_gen.samples}")
print(f"Test samples     : {test_gen.samples}")
print()


# ─────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────

print("Loading CNN model...")
model = load_model(MODEL_PATH)
print("Model loaded ✅")
print()


# ─────────────────────────────────────────────────────────
# Get predictions
# ─────────────────────────────────────────────────────────

def get_predictions(generator):
    generator.reset()
    y_prob = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = generator.classes
    return y_true, y_pred, y_prob


print("Getting predictions on train set...")
y_true_train, y_pred_train, y_prob_train = get_predictions(train_gen)

print("Getting predictions on val set...")
y_true_val, y_pred_val, y_prob_val = get_predictions(val_gen)

print("Getting predictions on test set...")
y_true_test, y_pred_test, y_prob_test = get_predictions(test_gen)

classes = list(train_gen.class_indices.keys())
print()


# ─────────────────────────────────────────────────────────
# Compute all metrics
# ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, split_name):
    print("=" * 60)
    print(f"  {split_name.upper()} SET METRICS")
    print("=" * 60)

    # ── Macro metrics ──────────────────────────────────────
    macro_precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # ── Micro metrics ──────────────────────────────────────
    micro_precision = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    micro_recall = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    micro_f1 = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    # ── AUC-ROC and AUPRC ─────────────────────────────────
    # Binarize labels for multiclass AUC
    classes_list = list(range(NUM_CLASSES))
    y_true_bin = label_binarize(y_true, classes=classes_list)

    try:
        auc_roc = roc_auc_score(
            y_true_bin, y_prob,
            average="macro",
            multi_class="ovr"
        )
    except Exception as e:
        auc_roc = float("nan")
        print(f"  AUC-ROC error: {e}")

    try:
        auprc = average_precision_score(
            y_true_bin, y_prob,
            average="macro"
        )
    except Exception as e:
        auprc = float("nan")
        print(f"  AUPRC error: {e}")

    # ── Accuracy ───────────────────────────────────────────
    accuracy = np.mean(y_true == y_pred)

    # ── Print results ──────────────────────────────────────
    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Accuracy':<30} {accuracy:>10.4f}")
    print(f"  {'-'*42}")
    print(f"  {'Macro Precision':<30} {macro_precision:>10.4f}")
    print(f"  {'Macro Recall':<30} {macro_recall:>10.4f}")
    print(f"  {'Macro F1':<30} {macro_f1:>10.4f}")
    print(f"  {'-'*42}")
    print(f"  {'Micro Precision':<30} {micro_precision:>10.4f}")
    print(f"  {'Micro Recall':<30} {micro_recall:>10.4f}")
    print(f"  {'Micro F1':<30} {micro_f1:>10.4f}")
    print(f"  {'-'*42}")
    print(f"  {'AUC-ROC (macro OvR)':<30} {auc_roc:>10.4f}")
    print(f"  {'AUPRC (macro)':<30} {auprc:>10.4f}")
    print()

    return {
        "accuracy"        : round(float(accuracy),        4),
        "macro_precision" : round(float(macro_precision),  4),
        "macro_recall"    : round(float(macro_recall),     4),
        "macro_f1"        : round(float(macro_f1),         4),
        "micro_precision" : round(float(micro_precision),  4),
        "micro_recall"    : round(float(micro_recall),     4),
        "micro_f1"        : round(float(micro_f1),         4),
        "auc_roc"         : round(float(auc_roc),          4),
        "auprc"           : round(float(auprc),            4),
    }


# ─────────────────────────────────────────────────────────
# Run for all splits
# ─────────────────────────────────────────────────────────

train_metrics = compute_metrics(
    y_true_train, y_pred_train, y_prob_train, "Train"
)
val_metrics = compute_metrics(
    y_true_val, y_pred_val, y_prob_val, "Validation"
)
test_metrics = compute_metrics(
    y_true_test, y_pred_test, y_prob_test, "Test"
)


# ─────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("  SUMMARY — ALL SPLITS")
print("=" * 60)
print(f"\n  {'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
print(f"  {'-'*57}")

metrics_to_show = [
    ("Accuracy",         "accuracy"),
    ("Macro Precision",  "macro_precision"),
    ("Macro Recall",     "macro_recall"),
    ("Macro F1",         "macro_f1"),
    ("Micro Precision",  "micro_precision"),
    ("Micro Recall",     "micro_recall"),
    ("Micro F1",         "micro_f1"),
    ("AUC-ROC",          "auc_roc"),
    ("AUPRC",            "auprc"),
]

for label, key in metrics_to_show:
    t = train_metrics[key]
    v = val_metrics[key]
    te = test_metrics[key]
    print(f"  {label:<25} {t:>10.4f} {v:>10.4f} {te:>10.4f}")

print()


# ─────────────────────────────────────────────────────────
# Save to JSON
# ─────────────────────────────────────────────────────────

all_metrics = {
    "model"      : BASE_MODEL,
    "train"      : train_metrics,
    "validation" : val_metrics,
    "test"       : test_metrics
}

output_path = Path("metrics/cnn_comprehensive_metrics.json")
with open(output_path, "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"All metrics saved → {output_path}")