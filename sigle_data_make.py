#!/usr/bin/env python3
"""
Script to create single-image datasets from nuScenes dataset.
Replaces all images with a single source image while keeping all other data unchanged.
"""

import os
import shutil
from pathlib import Path


def create_single_image_dataset(
    source_image_path: str,
    dataset_name: str,
    base_dataset_path: str = "/usst906disk/yyh/datasets/nuScenes",
    output_base_path: str = "/usst906disk/yyh/datasets",
) -> str:
    """
    Create a new dataset by replacing all images with a single source image.

    Args:
        source_image_path: Path to the source image (e.g., pic_sample/000.png)
        dataset_name: Name of the new dataset (e.g., nuScenes_sigle_1)
        base_dataset_path: Path to the original nuScenes dataset
        output_base_path: Base path for output datasets

    Returns:
        Path to the created dataset
    """
    source_image = Path(source_image_path)
    output_path = Path(output_base_path) / dataset_name
    base_dataset = Path(base_dataset_path)

    print(f"Creating single-image dataset: {dataset_name}")
    print(f"Source image: {source_image}")
    print(f"Output path: {output_path}")

    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")

    if not base_dataset.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_dataset}")

    if output_path.exists():
        print(f"Removing existing dataset at: {output_path}")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_count = 0
    other_count = 0

    print("\nCopying and replacing images...")
    for root, dirs, files in os.walk(base_dataset):
        rel_path = Path(root).relative_to(base_dataset)
        dest_dir = output_path / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_file = Path(root) / file
            dest_file = dest_dir / file

            if Path(file).suffix.lower() in image_extensions:
                shutil.copy2(source_image, dest_file)
                image_count += 1
                if image_count % 500 == 0:
                    print(f"  Processed {image_count} images...")
            else:
                shutil.copy2(src_file, dest_file)
                other_count += 1

    print(f"\nCompleted!")
    print(f"  Images replaced: {image_count}")
    print(f"  Other files copied: {other_count}")
    print(f"  Dataset saved to: {output_path}")

    return str(output_path)


def main():
    source_image = "/usst906disk/yyh/project/FSdrive_github/FSDrive/pic_sample/000.png"

    create_single_image_dataset(
        source_image_path=source_image,
        dataset_name="nuScenes_sigle_1",
    )


if __name__ == "__main__":
    main()
