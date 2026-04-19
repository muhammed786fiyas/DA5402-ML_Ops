# src/prepare.py
# Assisted by Claude

import os
import random
import argparse
from pathlib import Path
from PIL import Image

from src.utils.logger import get_logger
from src.utils.config import load_config, load_params

logger = get_logger("prepare")


def get_valid_breeds(raw_dir: Path, min_images: int) -> list:
    """
    Return list of (folder, images) tuples for breeds
    that have >= min_images images.
    Skips folders with fewer than min_images (assignment requirement).
    """
    valid_breeds = []

    for folder in sorted(raw_dir.iterdir()):
        if not folder.is_dir():
            logger.debug(f"Skipping non-folder item: {folder.name}")
            continue

        # Count image files only
        images = [
            f for f in folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
        count = len(images)

        if count < min_images:
            logger.warning(
                f"Skipping '{folder.name}' — only {count} image(s) "
                f"(minimum required: {min_images})"
            )
        else:
            logger.debug(f"Valid breed: '{folder.name}' — {count} images")
            valid_breeds.append((folder, images))

    return valid_breeds


def resize_and_copy(image_path: Path, dest_path: Path, img_size: int) -> bool:
    """
    Resize a single image and save to destination.
    Returns True if successful, False if failed.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")   # handles grayscale or RGBA
            img = img.resize((img_size, img_size), Image.LANCZOS)
            img.save(dest_path)
        return True
    except Exception as e:
        logger.error(
            f"Failed to process '{image_path.name}': {e}",
            exc_info=True
        )
        return False


def split_images(images: list, test_split: float, val_split: float, seed: int):
    """
    Split image list into train, val, test sets.
    Golden split: 10% test, 10% val, 80% train (assignment requirement).
    """
    random.seed(seed)
    random.shuffle(images)

    total      = len(images)
    test_count = max(1, int(total * test_split))
    val_count  = max(1, int(total * val_split))

    test  = images[:test_count]
    val   = images[test_count: test_count + val_count]
    train = images[test_count + val_count:]

    return train, val, test


def prepare():
    logger.info("=" * 55)
    logger.info("PREPARE STAGE STARTED")
    logger.info("=" * 55)

    # ── Load config and params ───────────────────────────────
    config = load_config()
    params = load_params()

    # Paths from config.yaml
    raw_dir    = config["paths"]["raw_dir"]
    output_dir = config["paths"]["processed_dir"]

    # Hyperparameters from params.yaml
    img_size   = params["prepare"]["img_size"]
    test_split = params["prepare"]["test_split"]
    val_split  = params["prepare"]["val_split"]
    seed       = params["prepare"]["seed"]
    min_images = params["prepare"]["min_images"]

    logger.info(f"Paths loaded from config.yaml")
    logger.info(f"  raw_dir    → {raw_dir}")
    logger.info(f"  output_dir → {output_dir}")
    logger.info(
        f"Parameters loaded from params.yaml → "
        f"img_size={img_size}, test_split={test_split}, "
        f"val_split={val_split}, seed={seed}, min_images={min_images}"
    )

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    # Validate raw data exists
    if not raw_path.exists():
        logger.error(f"Raw data directory not found: {raw_path}")
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    # ── Step 1: Find valid breeds ────────────────────────────
    logger.info(f"Step 1: Scanning raw data → {raw_path}")
    valid_breeds = get_valid_breeds(raw_path, min_images)

    total_folders = sum(1 for f in raw_path.iterdir() if f.is_dir())
    logger.info(
        f"Found {total_folders} total folders — "
        f"{len(valid_breeds)} valid breeds (>= {min_images} images)"
    )

    if len(valid_breeds) == 0:
        logger.error("No valid breed folders found. Exiting.")
        raise ValueError("No valid breed folders found.")

    # ── Step 2: Create output directories ───────────────────
    logger.info("Step 2: Creating output directory structure")
    for split in ["train", "val", "test"]:
        for folder, _ in valid_breeds:
            (out_path / split / folder.name).mkdir(
                parents=True, exist_ok=True
            )
    logger.debug(f"Output directories created under: {out_path}")

    # ── Step 3: Resize + Split ───────────────────────────────
    logger.info(
        f"Step 3: Resizing to {img_size}x{img_size} and splitting data"
    )

    total_train     = 0
    total_val       = 0
    total_test      = 0
    total_processed = 0
    total_failed    = 0
    total_images    = sum(len(imgs) for _, imgs in valid_breeds)

    for folder, images in valid_breeds:
        breed_name = folder.name
        logger.debug(
            f"Processing breed: '{breed_name}' ({len(images)} images)"
        )

        train_imgs, val_imgs, test_imgs = split_images(
            images, test_split, val_split, seed
        )

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val",   val_imgs),
            ("test",  test_imgs),
        ]:
            dest_dir = out_path / split_name / breed_name
            for img_path in split_imgs:
                dest_path = dest_dir / img_path.name
                success = resize_and_copy(img_path, dest_path, img_size)

                if success:
                    total_processed += 1
                else:
                    total_failed += 1

                # Progress update every 100 images
                done = total_processed + total_failed
                if done % 100 == 0:
                    logger.info(
                        f"Progress: {done}/{total_images} images processed"
                    )

        total_train += len(train_imgs)
        total_val   += len(val_imgs)
        total_test  += len(test_imgs)

        logger.debug(
            f"  '{breed_name}' split → "
            f"train={len(train_imgs)}, "
            f"val={len(val_imgs)}, "
            f"test={len(test_imgs)}"
        )

    # ── Step 4: Final summary ────────────────────────────────
    logger.info("Step 4: Final summary")
    logger.info(f"  Total images processed : {total_processed}")
    logger.info(f"  Failed images          : {total_failed}")
    logger.info(f"  Train set size         : {total_train}")
    logger.info(f"  Val   set size         : {total_val}")
    logger.info(f"  Test  set size         : {total_test}")
    logger.info(f"  Number of classes      : {len(valid_breeds)}")
    logger.info(f"  Output saved to        : {out_path}")

    logger.info("=" * 55)
    logger.info("PREPARE STAGE COMPLETE ✓")
    logger.info("=" * 55)


if __name__ == "__main__":
    prepare()