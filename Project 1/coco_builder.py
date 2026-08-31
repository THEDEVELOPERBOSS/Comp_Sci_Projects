import json
import random
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from downloader import ensure_dataset_files


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"

ANNOTATIONS_DIR = DATASET_DIR / "annotations"

CLASSIFICATION_DIR = DATASET_DIR / "classification"

TEMP_DIR = DATASET_DIR / "temp_coco_images"


# ============================================================
# SETTINGS
# ============================================================

MAX_IMAGES_PER_CLASS = 1000

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

MIN_OBJECT_SIZE = 100
MIN_OBJECT_AREA_RATIO = 0.02

RANDOM_SEED = 42


# ============================================================
# COCO CLASSES
# ============================================================

COCO_CLASSES = {
    "backpack": "backpack",
    "laptop": "laptop",
    "cell phone": "phone",
    "dining table": "table",
    "couch": "couch",
    "bottle": "bottle",
    "car": "car",
    "truck": "truck",
}


# ============================================================
# DATASET STATUS
# ============================================================

def get_class_images(class_name):
    """
    Find existing images for a class.

    This checks the old flat structure:
        classification/class_name/

    It also checks the final split structure.
    """

    images = []

    # Old/current structure
    old_folder = CLASSIFICATION_DIR / class_name

    if old_folder.exists():

        images.extend(
            old_folder.glob("*.jpg")
        )

        images.extend(
            old_folder.glob("*.jpeg")
        )

        images.extend(
            old_folder.glob("*.png")
        )

    # Final structure
    for split in ["train", "val", "test"]:

        folder = (
            CLASSIFICATION_DIR
            / split
            / class_name
        )

        if folder.exists():

            images.extend(
                folder.glob("*.jpg")
            )

            images.extend(
                folder.glob("*.jpeg")
            )

            images.extend(
                folder.glob("*.png")
            )

    return images


def count_class_images(class_name):

    return len(
        get_class_images(class_name)
    )


def print_dataset_status():

    print("\nCurrent dataset:")

    complete = True

    for class_name in COCO_CLASSES.values():

        count = count_class_images(
            class_name
        )

        print(
            f"  {class_name:<15}"
            f"{count}/{MAX_IMAGES_PER_CLASS}"
        )

        if count < MAX_IMAGES_PER_CLASS:
            complete = False

    return complete


def dataset_is_complete():

    for class_name in COCO_CLASSES.values():

        if (
            count_class_images(class_name)
            < MAX_IMAGES_PER_CLASS
        ):

            return False

    return True


def dataset_is_split():

    """
    Check whether the dataset has already
    been divided into train/val/test.
    """

    for class_name in COCO_CLASSES.values():

        for split in [
            "train",
            "val",
            "test"
        ]:

            folder = (
                CLASSIFICATION_DIR
                / split
                / class_name
            )

            if not folder.exists():
                return False

            if not any(
                folder.iterdir()
            ):
                return False

    return True


# ============================================================
# SPLIT EXISTING DATASET
# ============================================================

def split_existing_dataset():

    """
    Take the existing flat dataset:

        classification/
            backpack/
            laptop/
            ...

    and split it into:

        classification/
            train/
            val/
            test/

    Images are MOVED, not copied.
    """

    print(
        "\n" +
        "=" * 60
    )

    print(
        "SPLITTING EXISTING DATASET"
    )

    print(
        "=" * 60
    )

    random.seed(
        RANDOM_SEED
    )

    for class_name in COCO_CLASSES.values():

        source_folder = (
            CLASSIFICATION_DIR
            / class_name
        )

        if not source_folder.exists():

            print(
                f"[WARNING] "
                f"{class_name} folder "
                f"not found."
            )

            continue

        images = []

        images.extend(
            source_folder.glob("*.jpg")
        )

        images.extend(
            source_folder.glob("*.jpeg")
        )

        images.extend(
            source_folder.glob("*.png")
        )

        if not images:

            print(
                f"[WARNING] "
                f"No images found for "
                f"{class_name}."
            )

            continue

        random.shuffle(
            images
        )

        # Only use the requested number.
        images = images[
            :MAX_IMAGES_PER_CLASS
        ]

        total = len(images)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        train_images = images[
            :train_count
        ]

        val_images = images[
            train_count:
            train_count + val_count
        ]

        test_images = images[
            train_count + val_count:
        ]

        split_groups = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for (
            split_name,
            split_images
        ) in split_groups.items():

            destination_folder = (
                CLASSIFICATION_DIR
                / split_name
                / class_name
            )

            destination_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            for image_path in split_images:

                destination = (
                    destination_folder
                    / image_path.name
                )

                if destination.exists():
                    continue

                shutil.move(
                    str(image_path),
                    str(destination)
                )

        print(
            f"[OK] {class_name}: "
            f"{len(train_images)} train, "
            f"{len(val_images)} val, "
            f"{len(test_images)} test"
        )

        # Remove empty source folder.
        try:

            source_folder.rmdir()

        except OSError:

            pass

    print(
        "\n[OK] Existing dataset split."
    )


# ============================================================
# ANNOTATIONS
# ============================================================

def extract_annotations():

    annotation_zip = (
        ANNOTATIONS_DIR
        / "annotations_trainval2017.zip"
    )

    train_json = (
        ANNOTATIONS_DIR
        / "instances_train2017.json"
    )

    val_json = (
        ANNOTATIONS_DIR
        / "instances_val2017.json"
    )

    if (
        train_json.exists()
        and val_json.exists()
    ):

        return train_json, val_json

    if not annotation_zip.exists():

        print(
            "[ERROR] COCO annotation ZIP "
            "not found."
        )

        return None, None

    print(
        "\nExtracting COCO annotations..."
    )

    with zipfile.ZipFile(
        annotation_zip,
        "r"
    ) as archive:

        for member in archive.namelist():

            if (
                member.endswith(
                    "instances_train2017.json"
                )
                or
                member.endswith(
                    "instances_val2017.json"
                )
            ):

                target = (
                    ANNOTATIONS_DIR
                    / Path(member).name
                )

                if target.exists():
                    continue

                with archive.open(
                    member
                ) as source:

                    with open(
                        target,
                        "wb"
                    ) as destination:

                        shutil.copyfileobj(
                            source,
                            destination
                        )

    return train_json, val_json


# ============================================================
# COCO DOWNLOAD
# ============================================================

def download_image(
    image_info,
    split
):

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_name = image_info["file_name"]

    image_path = (
        TEMP_DIR
        / file_name
    )

    if image_path.exists():

        return image_path

    if split == "train":

        url = (
            "http://images.cocodataset.org/"
            f"train2017/{file_name}"
        )

    else:

        url = (
            "http://images.cocodataset.org/"
            f"val2017/{file_name}"
        )

    try:

        response = requests.get(
            url,
            stream=True,
            timeout=60
        )

        response.raise_for_status()

        with open(
            image_path,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    file.write(chunk)

        return image_path

    except Exception as error:

        print(
            f"\nDownload failed: "
            f"{error}"
        )

        return None


# ============================================================
# CROP
# ============================================================

def crop_object(
    image_path,
    annotation,
    output_path
):

    try:

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            image_width, image_height = (
                image.size
            )

            x, y, width, height = (
                annotation["bbox"]
            )

            if (
                width < MIN_OBJECT_SIZE
                or height < MIN_OBJECT_SIZE
            ):

                return False

            image_area = (
                image_width
                * image_height
            )

            object_area = (
                width * height
            )

            if (
                object_area / image_area
                < MIN_OBJECT_AREA_RATIO
            ):

                return False

            left = max(
                0,
                int(x)
            )

            top = max(
                0,
                int(y)
            )

            right = min(
                image_width,
                int(x + width)
            )

            bottom = min(
                image_height,
                int(y + height)
            )

            if (
                right <= left
                or bottom <= top
            ):

                return False

            crop = image.crop(
                (
                    left,
                    top,
                    right,
                    bottom
                )
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            crop.save(
                output_path,
                "JPEG",
                quality=95
            )

            return True

    except Exception:

        return False


# ============================================================
# COLLECT MISSING IMAGES
# ============================================================

def collect_missing_images(
    class_name,
    annotations,
    image_lookup,
    split,
    needed
):

    if needed <= 0:
        return 0

    output_folder = (
        CLASSIFICATION_DIR
        / split
        / class_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    random.shuffle(
        annotations
    )

    saved = 0

    for annotation in tqdm(
        annotations,
        desc=f"{class_name} [{split}]"
    ):

        if saved >= needed:
            break

        image_info = (
            image_lookup.get(
                annotation["image_id"]
            )
        )

        if image_info is None:
            continue

        output_file = (
            output_folder
            / (
                f"{annotation['image_id']}_"
                f"{annotation['id']}.jpg"
            )
        )

        if output_file.exists():
            continue

        image_path = download_image(
            image_info,
            split
        )

        if image_path is None:
            continue

        success = crop_object(
            image_path,
            annotation,
            output_file
        )

        try:

            if image_path.exists():
                image_path.unlink()

        except Exception:
            pass

        if success:
            saved += 1

    return saved


# ============================================================
# BUILD DATASET
# ============================================================

def build_coco_dataset():

    print(
        "\n" +
        "=" * 60
    )

    print(
        "COCO DATASET CHECK"
    )

    print(
        "=" * 60
    )

    complete = print_dataset_status()

    # --------------------------------------------------------
    # DATASET EXISTS BUT IS NOT SPLIT
    # --------------------------------------------------------

    if complete and not dataset_is_split():

        print(
            "\n[OK] You already have all "
            "8,000 images."
        )

        print(
            "No downloads are necessary."
        )

        split_existing_dataset()

        print_dataset_status()

        return

    # --------------------------------------------------------
    # DATASET ALREADY COMPLETELY SPLIT
    # --------------------------------------------------------

    if complete and dataset_is_split():

        print(
            "\n[OK] Dataset is already "
            "complete and organized."
        )

        print(
            "No downloads are necessary."
        )

        return

    # --------------------------------------------------------
    # SOME DATA IS MISSING
    # --------------------------------------------------------

    print(
        "\n[INFO] Some images are missing."
    )

    print(
        "Existing images will be kept."
    )

    downloads_complete = (
        ensure_dataset_files()
    )

    if not downloads_complete:

        print(
            "\n[ERROR] Required files "
            "are not available."
        )

        return

    train_json, val_json = (
        extract_annotations()
    )

    if (
        train_json is None
        or val_json is None
    ):

        return

    # --------------------------------------------------------
    # COLLECT MISSING DATA
    # --------------------------------------------------------

    for split, json_path in [
        ("train", train_json),
        ("val", val_json)
    ]:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        image_lookup = {
            image["id"]: image
            for image in data["images"]
        }

        category_lookup = {
            category["id"]:
            category["name"]
            for category in data["categories"]
        }

        annotations_by_category = (
            defaultdict(list)
        )

        for annotation in data["annotations"]:

            category_name = (
                category_lookup.get(
                    annotation["category_id"]
                )
            )

            if category_name in COCO_CLASSES:

                annotations_by_category[
                    category_name
                ].append(annotation)

        for (
            coco_name,
            class_name
        ) in COCO_CLASSES.items():

            current_count = (
                count_class_images(
                    class_name
                )
            )

            needed = (
                MAX_IMAGES_PER_CLASS
                - current_count
            )

            if needed <= 0:
                continue

            print(
                f"\n{class_name}: "
                f"{current_count}/"
                f"{MAX_IMAGES_PER_CLASS}"
            )

            collect_missing_images(
                class_name,
                annotations_by_category[
                    coco_name
                ],
                image_lookup,
                split,
                needed
            )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    if dataset_is_complete():

        split_existing_dataset()

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    if TEMP_DIR.exists():

        try:
            shutil.rmtree(TEMP_DIR)
        except Exception:
            pass

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print(
        "\n" +
        "=" * 60
    )

    print(
        "FINAL DATASET STATUS"
    )

    print(
        "=" * 60
    )

    print_dataset_status()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    build_coco_dataset()