
from pathlib import Path

# Folder containing the extracted image crops
DATASET_DIR = Path(__file__).resolve().parent / "open_images_crops"

# Image formats we want to count
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

total_images = 0

for split in ("train", "validation", "test"):
    split_dir = DATASET_DIR / split
    print(f"\n{split.upper()}")

    split_total = 0

    if not split_dir.exists():
        print("  Split folder not found.")
        continue

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"  {class_dir.name}: {count:,}")
        split_total += count

    print(f"  Total: {split_total:,}")
    total_images += split_total

print(f"\nTOTAL IMAGES: {total_images:,}")
