
import fiftyone as fo

DATASET_NAME = "open_images_classifier"

# Open the dataset already downloaded
dataset = fo.load_dataset(DATASET_NAME)

print(f"Dataset: {dataset.name}")
print(f"Total images: {len(dataset)}")

# Show the fields available on each image
print("\nAvailable fields:")
for field in dataset.get_field_schema():
    print(f"  {field}")

# Inspect one image to find its annotations and split
sample = dataset.first()

if sample is not None:
    print("\nFirst image:")
    print(f"File: {sample.filepath}")
    print(f"Split tags: {sample.tags}")

    for field in dataset.get_field_schema():
        value = sample[field]
        if value is not None and field != "filepath":
            print(f"\n{field}:")
            print(str(value)[:1500])
