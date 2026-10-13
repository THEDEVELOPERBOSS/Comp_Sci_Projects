
import fiftyone as fo

DATASET_NAME = "open_images_classifier"

dataset = fo.load_dataset(DATASET_NAME)

split_counts = {
    "train": 0,
    "validation": 0,
    "test": 0,
    "other": 0,
}

for sample in dataset.iter_samples(progress=True):
    tags = sample.tags

    if "train" in tags:
        split_counts["train"] += 1
    elif "validation" in tags:
        split_counts["validation"] += 1
    elif "test" in tags:
        split_counts["test"] += 1
    else:
        split_counts["other"] += 1

print("\nOriginal source split counts:")

for split, count in split_counts.items():
    print(f"{split}: {count:,}")
