# src/transform.py
# Assisted by Claude

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import random
import shutil

from skimage.feature import hog
from skimage import exposure

from src.utils.logger import get_logger
from src.utils.config import load_config, load_params

logger = get_logger("transform")


# ─────────────────────────────────────────────────────────
# Image Augmentation Helpers
# ─────────────────────────────────────────────────────────

def augment_image(img: Image.Image, seed: int = 42) -> Image.Image:
    """
    Apply random augmentations to a PIL image.
    Augmentations: horizontal flip, rotation, brightness, contrast.
    These are applied randomly to increase dataset diversity.
    """
    random.seed(seed)

    # Horizontal flip (50% chance)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        logger.debug("Applied horizontal flip")

    # Random rotation between -15 and +15 degrees
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    logger.debug(f"Applied rotation: {angle:.2f} degrees")

    # Random brightness (0.8 to 1.2)
    brightness_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    logger.debug(f"Applied brightness: {brightness_factor:.2f}")

    # Random contrast (0.8 to 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    logger.debug(f"Applied contrast: {contrast_factor:.2f}")

    return img


# ─────────────────────────────────────────────────────────
# HOG Feature Extraction
# ─────────────────────────────────────────────────────────

def extract_hog_features(img: Image.Image, params: dict) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from image.
    HOG captures edge and texture information — great for biometric identification.

    Args:
        img    : PIL Image (RGB, already resized)
        params : transform params from params.yaml

    Returns:
        1D numpy array of HOG features
    """
    orientations      = params["hog_orientations"]
    pixels_per_cell   = params["hog_pixels_per_cell"]
    cells_per_block   = params["hog_cells_per_block"]

    # Convert to grayscale for HOG
    img_gray = np.array(img.convert("L"))

    features = hog(
        img_gray,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        visualize=False
    )

    return features


def load_image(image_path: Path) -> Image.Image:
    """Load and return a PIL image in RGB format."""
    with Image.open(image_path) as img:
        return img.convert("RGB").copy()


# ─────────────────────────────────────────────────────────
# Build Augmented Dataset (v2)
# ─────────────────────────────────────────────────────────

def create_augmented_dataset(
    v1_dir: Path,
    v2_dir: Path,
    seed: int
):
    """
    Create v2_augmented dataset from v1_resized.
    For train split: copies original + adds augmented copy.
    For val/test splits: copies as-is (NEVER augment val/test).
    """
    logger.info("Creating v2_augmented dataset from v1_resized")

    total_created = 0

    for split in ["train", "val", "test"]:
        split_src = v1_dir / split
        split_dst = v2_dir / split

        if not split_src.exists():
            logger.warning(f"Split folder not found: {split_src}")
            continue

        for breed_folder in sorted(split_src.iterdir()):
            if not breed_folder.is_dir():
                continue

            breed_name = breed_folder.name
            dest_breed  = split_dst / breed_name
            dest_breed.mkdir(parents=True, exist_ok=True)

            images = list(breed_folder.iterdir())

            for img_path in images:
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue

                try:
                    img = load_image(img_path)

                    # Always copy original image
                    dest_orig = dest_breed / img_path.name
                    img.save(dest_orig)
                    total_created += 1

                    # Only augment training data — NEVER val or test
                    if split == "train":
                        aug_img  = augment_image(img.copy(), seed=seed)
                        aug_name = f"aug_{img_path.name}"
                        aug_img.save(dest_breed / aug_name)
                        total_created += 1

                except Exception as e:
                    logger.error(
                        f"Failed to process '{img_path.name}': {e}",
                        exc_info=True
                    )

        logger.info(
            f"  {split} split → "
            f"{'original + augmented' if split == 'train' else 'original only (no augmentation)'}"
        )

    logger.info(f"v2_augmented total images created: {total_created}")
    return total_created


# ─────────────────────────────────────────────────────────
# Extract HOG Features for All Splits
# ─────────────────────────────────────────────────────────

def extract_features_for_split(
    split_dir: Path,
    split_name: str,
    params: dict,
    features_dir: Path
):
    """
    Extract HOG features from all images in a split folder.
    Saves features and labels as .npy files.

    Args:
        split_dir    : path to split folder (train/val/test)
        split_name   : name of split for saving files
        params       : transform params from params.yaml
        features_dir : where to save .npy files
    """
    features_list = []
    labels_list   = []
    failed        = 0

    # Get sorted breed folders for consistent label encoding
    breed_folders = sorted([
        f for f in split_dir.iterdir() if f.is_dir()
    ])

    # Build label map: breed_name → integer
    label_map = {breed.name: idx for idx, breed in enumerate(breed_folders)}
    logger.info(f"  Label map: {label_map}")

    total_images = sum(
        len(list(b.iterdir())) for b in breed_folders
    )
    processed = 0

    for breed_folder in breed_folders:
        breed_name  = breed_folder.name
        label       = label_map[breed_name]

        for img_path in sorted(breed_folder.iterdir()):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            try:
                img      = load_image(img_path)
                features = extract_hog_features(img, params)
                features_list.append(features)
                labels_list.append(label)
                processed += 1

                if processed % 200 == 0:
                    logger.info(
                        f"  [{split_name}] Progress: "
                        f"{processed}/{total_images} images"
                    )

            except Exception as e:
                logger.error(
                    f"HOG extraction failed for '{img_path.name}': {e}",
                    exc_info=True
                )
                failed += 1

    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)

    # Save features and labels
    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_dir / f"hog_{split_name}.npy", X)
    np.save(features_dir / f"labels_{split_name}.npy", y)

    logger.info(
        f"  [{split_name}] Feature shape: {X.shape} | "
        f"Labels shape: {y.shape} | Failed: {failed}"
    )

    return X.shape, failed


# ─────────────────────────────────────────────────────────
# Save Label Map
# ─────────────────────────────────────────────────────────

def save_label_map(v2_train_dir: Path, features_dir: Path):
    """
    Save label map (breed name → index) as a .npy file.
    Needed during evaluation to decode predictions back to breed names.
    """
    breed_folders = sorted([
        f for f in v2_train_dir.iterdir() if f.is_dir()
    ])
    label_map = {breed.name: idx for idx, breed in enumerate(breed_folders)}

    np.save(features_dir / "label_map.npy", label_map)
    logger.info(f"Label map saved → {features_dir / 'label_map.npy'}")
    logger.info(f"Classes: {list(label_map.keys())}")

    return label_map


# ─────────────────────────────────────────────────────────
# Main Transform Function
# ─────────────────────────────────────────────────────────

def transform():
    logger.info("=" * 55)
    logger.info("TRANSFORM STAGE STARTED")
    logger.info("=" * 55)

    # ── Load config and params ───────────────────────────────
    config = load_config()
    params = load_params()

    # Paths from config.yaml
    v1_dir       = Path(config["paths"]["processed_dir"])     # data/processed/v1_resized
    v2_dir       = Path(config["paths"]["augmented_dir"])     # data/processed/v2_augmented
    features_dir = Path(config["paths"]["features_dir"])      # data/features

    # Params from params.yaml
    transform_params = params["transform"]
    seed             = params["prepare"]["seed"]

    logger.info(f"Paths loaded from config.yaml")
    logger.info(f"  v1_resized   → {v1_dir}")
    logger.info(f"  v2_augmented → {v2_dir}")
    logger.info(f"  features_dir → {features_dir}")
    logger.info(
        f"Transform params → "
        f"method={transform_params['method']}, "
        f"orientations={transform_params['hog_orientations']}, "
        f"pixels_per_cell={transform_params['hog_pixels_per_cell']}, "
        f"cells_per_block={transform_params['hog_cells_per_block']}"
    )

    # Validate v1 exists
    if not v1_dir.exists():
        logger.error(f"v1_resized not found at {v1_dir}. Run prepare.py first.")
        raise FileNotFoundError(f"v1_resized not found: {v1_dir}")

    # ── Step 1: Create v2_augmented dataset ─────────────────
    logger.info("Step 1: Creating v2_augmented dataset")
    total_created = create_augmented_dataset(v1_dir, v2_dir, seed)
    logger.info(f"v2_augmented created with {total_created} total images")

    # ── Step 2: Extract HOG features ────────────────────────
    logger.info("Step 2: Extracting HOG features from v2_augmented")

    for split in ["train", "val", "test"]:
        split_dir = v2_dir / split
        if not split_dir.exists():
            logger.warning(f"Split not found, skipping: {split_dir}")
            continue

        logger.info(f"Extracting HOG features for '{split}' split...")
        shape, failed = extract_features_for_split(
            split_dir, split, transform_params, features_dir
        )
        logger.info(
            f"  '{split}' done → shape={shape}, failed={failed}"
        )

    # ── Step 3: Save label map ───────────────────────────────
    logger.info("Step 3: Saving label map")
    label_map = save_label_map(v2_dir / "train", features_dir)

    # ── Step 4: Summary ─────────────────────────────────────
    logger.info("Step 4: Summary")
    logger.info(f"  v2_augmented saved to  : {v2_dir}")
    logger.info(f"  HOG features saved to  : {features_dir}")
    logger.info(f"  Number of classes      : {len(label_map)}")
    logger.info(f"  Feature files created  :")
    for f in sorted(features_dir.iterdir()):
        logger.info(f"    {f.name}")

    logger.info("=" * 55)
    logger.info("TRANSFORM STAGE COMPLETE ✓")
    logger.info("=" * 55)


if __name__ == "__main__":
    transform()