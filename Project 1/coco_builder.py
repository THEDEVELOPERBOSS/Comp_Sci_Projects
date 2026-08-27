import json
import random
import shutil
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

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = SCRIPT_DIR / "dataset"

ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"

RAW_DIR = OUTPUT_DIR / "raw"

FINAL_DIR = OUTPUT_DIR / "classification"


# ============================================================
# CONFIGURATION
# ============================================================

# Number of images to collect per class.
MAX_IMAGES_PER_CLASS = 1000

# Minimum object size in pixels.
MIN_OBJECT_SIZE = 100

# Minimum percentage of the original image
# occupied by the object.
MIN_OBJECT_AREA_RATIO = 0.02


# Dataset split percentages.
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# Make random selections reproducible.
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
# COCO URLS
# ============================================================

TRAIN_IMAGE_URL = (
    "http://images.cocodataset.org/train2017/"
    "{file_name}"
)

VAL_IMAGE_URL = (
    "http://images.cocodataset.org/val2017/"
    "{file_name}"
)


# ============================================================
# ANNOTATIONS
# ============================================================

def extract_annotations():
    """
    Extract the required COCO annotation JSON files.
    """

    annotation_zip = (
        ANNOTATIONS_DIR /
        "annotations_trainval2017.zip"
    )

    train_json = (
        ANNOTATIONS_DIR /
        "instances_train2017.json"
    )

    val_json = (
        ANNOTATIONS_DIR /
        "instances_val2017.json"
    )

    # If both files already exist,
    # there is nothing to do.
    if (
        train_json.exists()
        and val_json.exists()
    ):
        print(
            "[OK] Annotation files "
            "already extracted."
        )

        return train_json, val_json

    print(
        "\nExtracting annotation files..."
    )

    with zipfile.ZipFile(
        annotation_zip,
        "r"
    ) as zip_file:

        for member in zip_file.namelist():

            if (
                member.endswith(
                    "instances_train2017.json"
                )
                or member.endswith(
                    "instances_val2017.json"
                )
            ):

                target = (
                    ANNOTATIONS_DIR /
                    Path(member).name
                )

                if target.exists():
                    continue

                print(
                    f"Extracting "
                    f"{target.name}..."
                )

                with zip_file.open(
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
# LOAD COCO DATA
# ============================================================

def load_coco_annotations(
    json_path
):

    print(
        f"Loading "
        f"{json_path.name}..."
    )

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_image_lookup(
    data
):

    return {
        image["id"]: image
        for image in data["images"]
    }


def group_annotations_by_category(
    data
):

    category_lookup = {
        category["id"]:
        category["name"]

        for category in
        data["categories"]
    }

    grouped = defaultdict(list)

    for annotation in data["annotations"]:

        category_name = (
            category_lookup[
                annotation["category_id"]
            ]
        )

        if category_name in COCO_CLASSES:

            grouped[
                category_name
            ].append(
                annotation
            )

    return grouped


# ============================================================
# IMAGE DOWNLOAD
# ============================================================

def download_image(
    image_info,
    split
):
    """
    Download one COCO image to a temporary file.

    The image is deleted after its crop
    is successfully processed.
    """

    temp_dir = (
        OUTPUT_DIR /
        "temp_coco_images"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_name = (
        image_info["file_name"]
    )

    image_path = (
        temp_dir /
        file_name
    )

    if image_path.exists():

        return image_path

    if split == "train":

        url = (
            TRAIN_IMAGE_URL.format(
                file_name=file_name
            )
        )

    else:

        url = (
            VAL_IMAGE_URL.format(
                file_name=file_name
            )
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
            f"Failed to download "
            f"{file_name}: {error}"
        )

        return None


# ============================================================
# IMAGE CROPPING
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

            # Ignore very small objects.
            if width < MIN_OBJECT_SIZE:
                return False

            if height < MIN_OBJECT_SIZE:
                return False

            # Check how much of the image
            # the object occupies.
            image_area = (
                image_width *
                image_height
            )

            object_area = (
                width *
                height
            )

            area_ratio = (
                object_area /
                image_area
            )

            if (
                area_ratio <
                MIN_OBJECT_AREA_RATIO
            ):
                return False

            # Calculate crop coordinates.
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

            if right <= left:
                return False

            if bottom <= top:
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

    except Exception as error:

        print(
            f"Crop failed: "
            f"{error}"
        )

        return False


# ============================================================
# COLLECT ONE CLASS
# ============================================================

def collect_class_images(
    output_class,
    annotations,
    image_lookup,
    split
):

    class_dir = (
        RAW_DIR /
        output_class
    )

    class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Check how many images already exist.
    existing_images = list(
        class_dir.glob(
            "*.jpg"
        )
    )

    saved_count = len(
        existing_images
    )

    # If the class is already complete,
    # skip it.
    if (
        saved_count >=
        MAX_IMAGES_PER_CLASS
    ):

        print(
            f"[OK] "
            f"{output_class} "
            f"already has "
            f"{saved_count} images."
        )

        return

    random.shuffle(
        annotations
    )

    progress = tqdm(
        annotations,
        desc=(
            f"Collecting "
            f"{output_class}"
        )
    )

    for annotation in progress:

        if (
            saved_count >=
            MAX_IMAGES_PER_CLASS
        ):
            break

        image_id = (
            annotation["image_id"]
        )

        image_info = (
            image_lookup.get(
                image_id
            )
        )

        if image_info is None:
            continue

        output_file = (
            class_dir /
            f"{split}_"
            f"{image_id}_"
            f"{annotation['id']}.jpg"
        )

        # Don't recreate an existing crop.
        if output_file.exists():

            continue

        image_path = (
            download_image(
                image_info,
                split
            )
        )

        if image_path is None:

            continue

        success = crop_object(
            image_path,
            annotation,
            output_file
        )

        if success:

            saved_count += 1

        # IMPORTANT:
        # Delete the original COCO image
        # after the crop is finished.
        try:

            if image_path.exists():

                image_path.unlink()

        except Exception as error:

            print(
                f"Could not delete "
                f"temporary image: "
                f"{error}"
            )

        progress.set_postfix(
            saved=saved_count
        )

    print(
        f"\n{output_class}: "
        f"{saved_count} images"
    )


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset():
    """
    Move images into train,
    validation, and test folders.

    Files are MOVED instead of copied
    to avoid storing duplicates.
    """

    print(
        "\nSplitting dataset..."
    )

    random.seed(
        RANDOM_SEED
    )

    for class_folder in (
        RAW_DIR.iterdir()
    ):

        if not class_folder.is_dir():
            continue

        images = list(
            class_folder.glob(
                "*.jpg"
            )
        )

        random.shuffle(
            images
        )

        total = len(images)

        train_end = int(
            total *
            TRAIN_RATIO
        )

        val_end = (
            train_end +
            int(
                total *
                VAL_RATIO
            )
        )

        split_groups = {
            "train":
                images[:train_end],

            "val":
                images[
                    train_end:
                    val_end
                ],

            "test":
                images[
                    val_end:
                ],
        }

        for (
            split_name,
            split_images
        ) in split_groups.items():

            destination_dir = (
                FINAL_DIR /
                split_name /
                class_folder.name
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for image_path in split_images:

                destination = (
                    destination_dir /
                    image_path.name
                )

                # Move instead of copy.
                shutil.move(
                    str(image_path),
                    str(destination)
                )

        print(
            f"[OK] "
            f"{class_folder.name}: "
            f"{total} images split"
        )

    # Remove empty raw folders.
    try:

        if RAW_DIR.exists():

            shutil.rmtree(
                RAW_DIR
            )

            print(
                "[OK] Removed raw folder."
            )

    except Exception as error:

        print(
            f"Could not remove "
            f"raw folder: {error}"
        )


# ============================================================
# CLEANUP
# ============================================================

def cleanup_temp_images():

    temp_dir = (
        OUTPUT_DIR /
        "temp_coco_images"
    )

    if temp_dir.exists():

        try:

            shutil.rmtree(
                temp_dir
            )

            print(
                "[OK] Temporary COCO "
                "images deleted."
            )

        except Exception as error:

            print(
                f"Could not delete "
                f"temporary folder: "
                f"{error}"
            )


# ============================================================
# MAIN
# ============================================================

def build_coco_dataset():
   # --------------------------------------------------------
    # CHECK IF DATASET IS ALREADY COMPLETE
    # --------------------------------------------------------

    if dataset_is_complete():

        print(
            "\n" +
            "=" * 60
        )

        print(
            "COCO DATASET ALREADY COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            f"\nLocation:"
        )

        print(
            FINAL_DIR
        )

        return

    random.seed(
        RANDOM_SEED
    )

    print(
        "\nCOCO CLASSIFICATION "
        "DATASET BUILDER"
    )

    print(
        "\nClasses:"
    )

    for class_name in (
        COCO_CLASSES.values()
    ):

        print(
            f"  - {class_name}"
        )

    # --------------------------------------------------------
    # STEP 1:
    # CHECK/DOWNLOAD REQUIRED FILES
    # --------------------------------------------------------

    downloads_complete = (
        ensure_dataset_files()
    )

    if not downloads_complete:

        print(
            "\nDownloads are not complete."
        )

        print(
            "Run the program again "
            "to resume."
        )

        return

    # --------------------------------------------------------
    # STEP 2:
    # EXTRACT ANNOTATIONS
    # --------------------------------------------------------

    (
        train_json,
        val_json
    ) = extract_annotations()

    # --------------------------------------------------------
    # STEP 3:
    # LOAD AND PROCESS DATA
    # --------------------------------------------------------

    datasets = [
        (
            "train",
            train_json
        ),
        (
            "val",
            val_json
        ),
    ]

    for (
        split,
        json_path
    ) in datasets:

        data = (
            load_coco_annotations(
                json_path
            )
        )

        image_lookup = (
            build_image_lookup(
                data
            )
        )

        annotations_by_category = (
            group_annotations_by_category(
                data
            )
        )

        for (
            coco_class_name,
            output_class
        ) in COCO_CLASSES.items():

            annotations = (
                annotations_by_category[
                    coco_class_name
                ]
            )

            print(
                f"\nProcessing "
                f"{output_class}..."
            )

            collect_class_images(
                output_class,
                annotations,
                image_lookup,
                split
            )

    # --------------------------------------------------------
    # STEP 4:
    # SPLIT DATASET
    # --------------------------------------------------------

    split_dataset()

    # --------------------------------------------------------
    # STEP 5:
    # CLEANUP TEMPORARY FILES
    # --------------------------------------------------------

    cleanup_temp_images()

    print(
        "\n" +
        "=" * 60
    )

    print(
        "COCO DATASET BUILD COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nFinal dataset:"
    )

    print(
        FINAL_DIR
    )


# ============================================================
# RUN DIRECTLY
# ============================================================

def dataset_is_complete():
    """
    Check whether every class has the expected
    number of images in the final dataset.
    """

    if not FINAL_DIR.exists():
        return False

    for class_name in COCO_CLASSES.values():

        total_images = 0

        for split_name in [
            "train",
            "val",
            "test"
        ]:

            class_dir = (
                FINAL_DIR /
                split_name /
                class_name
            )

            if class_dir.exists():

                total_images += len(
                    list(
                        class_dir.glob(
                            "*.jpg"
                        )
                    )
                )

        if total_images < MAX_IMAGES_PER_CLASS:

            print(
                f"[INCOMPLETE] "
                f"{class_name}: "
                f"{total_images}/"
                f"{MAX_IMAGES_PER_CLASS}"
            )

            return False

    return True

if __name__ == "__main__":

    build_coco_dataset()