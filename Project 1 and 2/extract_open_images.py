
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Lock
from io import BytesIO
import shutil

import fiftyone as fo
from PIL import Image

# ============================================================
# Settings
# ============================================================

DATASET_NAME = "open_images_classifier"
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "open_images_crops"

CLASSES = {
    "Backpack": "backpack",
    "Bottle": "bottle",
    "Car": "car",
    "Couch": "couch",
    "Laptop": "laptop",
    "Mobile phone": "mobile_phone",
    "Table": "table",
    "Truck": "truck",
}

WORKERS = 4
MAX_PENDING = 8
MAX_OUTPUT_GB = None
MIN_FREE_GB = 10
MAX_IMAGE_SIZE = 512
JPEG_QUALITY = 88

MAX_OUTPUT_BYTES = (
    MAX_OUTPUT_GB * 1024**3
    if MAX_OUTPUT_GB is not None
    else None
)
MIN_FREE_BYTES = MIN_FREE_GB * 1024**3

# ============================================================
# Prepare folders and shared counters
# ============================================================

for split in ["train", "validation", "test"]:
    for folder_name in CLASSES.values():
        (OUTPUT_DIR / split / folder_name).mkdir(
            parents=True, exist_ok=True
        )


def get_folder_size(folder):
    total = 0
    for path in folder.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


output_bytes = get_folder_size(OUTPUT_DIR)
saved_count = 0
skipped_count = 0
unreadable_count = 0
counter_lock = Lock()



def save_crop(image, output_path):
    """Save one crop while protecting the remaining disk space."""
    global output_bytes

    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
    )
    data = buffer.getvalue()

    with counter_lock:
        # Do not overwrite an existing crop.
        if output_path.exists():
            return "skipped"

        # Stop before the drive runs too low on free space.
        free_bytes = shutil.disk_usage(OUTPUT_DIR.anchor).free

        if free_bytes - len(data) < MIN_FREE_BYTES:
            raise RuntimeError("Minimum free disk space reached.")

        # Save the crop.
        output_path.write_bytes(data)
        output_bytes += len(data)

    return "saved"


# ============================================================
# Worker: process one source image
# ============================================================

def process_image(job):
    global saved_count, skipped_count, unreadable_count

    filepath, split, sample_id, detections = job

    try:
        with Image.open(filepath) as original:
            image = original.convert("RGB")
    except (OSError, ValueError):
        with counter_lock:
            unreadable_count += 1
        return

    image_width, image_height = image.size

    local_saved = 0
    local_skipped = 0

    for label, detection_id, box, is_group in detections:
        if label not in CLASSES or is_group:
            continue

        x, y, width, height = box

        left = max(0, int(x * image_width))
        top = max(0, int(y * image_height))
        right = min(image_width, int((x + width) * image_width))
        bottom = min(image_height, int((y + height) * image_height))

        if right <= left or bottom <= top:
            continue

        folder = OUTPUT_DIR / split / CLASSES[label]
        output_path = folder / f"{sample_id}_{detection_id}.jpg"

        if output_path.exists():
            local_skipped += 1
            continue

        crop = image.crop((left, top, right, bottom))
        crop.thumbnail(
            (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )

        result = save_crop(crop, output_path)

        if result == "saved":
            local_saved += 1
        else:
            local_skipped += 1

    image.close()

    with counter_lock:
        saved_count += local_saved
        skipped_count += local_skipped


# ============================================================
# Main: coordinate dataset and worker threads
# ============================================================


def main():
    dataset = fo.load_dataset(DATASET_NAME)

    print(f"Dataset: {dataset.name}")
    print(f"Source images: {len(dataset)}")
    print(f"Workers: {WORKERS}")
    print(f"Existing output: {output_bytes / 1024**3:.2f} GB")
    print(f"Minimum free space: {MIN_FREE_GB} GB")

    pending = set()

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            # Process each split separately so training images
            # do not get processed before validation and test.
            for target_split in ("validation", "test", "train"):
                print(f"\nProcessing {target_split.upper()} split...")

                for sample in dataset.iter_samples(progress=True):
                    if target_split not in sample.tags:
                        continue

                    annotations = sample["ground_truth"]
                    if annotations is None:
                        continue

                    detections = [
                        (
                            detection.label,
                            detection.id,
                            tuple(detection.bounding_box),
                            getattr(detection, "IsGroupOf", False),
                        )
                        for detection in annotations.detections
                        if detection.label in CLASSES
                    ]

                    if not detections:
                        continue

                    job = (
                        sample.filepath,
                        target_split,
                        sample.id,
                        detections,
                    )

                    pending.add(executor.submit(process_image, job))

                    # Keep the number of queued jobs limited.
                    if len(pending) >= MAX_PENDING:
                        completed, pending = wait(
                            pending, return_when=FIRST_COMPLETED
                        )
                        for future in completed:
                            future.result()

                # Finish all jobs for this split before moving on.
                while pending:
                    completed, pending = wait(
                        pending, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        future.result()

                print(
                    f"Finished {target_split}. "
                    f"New crops saved so far: {saved_count:,}"
                )

    except KeyboardInterrupt:
        print("\nStop requested. Waiting for active workers to finish...")

    except RuntimeError as error:
        print(f"\nSafety stop: {error}")

    finally:
        print("\nExtraction summary")
        print(f"New crops saved: {saved_count:,}")
        print(f"Existing crops skipped: {skipped_count:,}")
        print(f"Unreadable images: {unreadable_count:,}")
        print(f"Output size: {output_bytes / 1024**3:.2f} GB")
        print(f"Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
