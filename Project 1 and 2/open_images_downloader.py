
import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = [
    "Backpack",
    "Bottle",
    "Car",
    "Couch",
    "Laptop",
    "Mobile phone",
    "Table",
    "Truck",
]

DATASET_NAME = "open_images_classifier"

try:
    # Reuse the existing dataset if it was created previously
    if fo.dataset_exists(DATASET_NAME):
        dataset = fo.load_dataset(DATASET_NAME)
        print(f"Resuming existing dataset: {DATASET_NAME}")

    else:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            splits=["train", "validation", "test"],
            label_types=["detections"],
            classes=CLASSES,
            only_matching=True,
            shuffle=True,
            seed=42,
            num_workers=4,
            dataset_name=DATASET_NAME,
            persistent=True,
        )

    print(f"\nDataset ready. Images: {len(dataset)}")

except KeyboardInterrupt:
    print("\nStop requested. Checking saved dataset...")

    if fo.dataset_exists(DATASET_NAME):
        dataset = fo.load_dataset(DATASET_NAME)
        print(f"Saved dataset contains {len(dataset)} images.")

    print("You can run this script again later.")
