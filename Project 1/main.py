import tensorflow as tf

from coco_builder import build_coco_dataset


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 32

EPOCHS = 10

TRAIN_DIR = "dataset/classification/train"

VAL_DIR = "dataset/classification/val"

TEST_DIR = "dataset/classification/test"

MODEL_PATH = "image_classifier.keras"


# ============================================================
# BUILD / CHECK DATASET
# ============================================================

build_coco_dataset()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_data = (
    tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=True,

        seed=42
    )
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

validation_data = (
    tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=False
    )
)


# ============================================================
# GET CLASS NAMES
# ============================================================

class_names = (
    train_data.class_names
)

NUM_CLASSES = len(
    class_names
)

print("\nClasses:")

for i, class_name in enumerate(
    class_names
):

    print(
        f"  {i}: {class_name}"
    )

print(
    f"\nNumber of classes: "
    f"{NUM_CLASSES}"
)


# ============================================================
# SPEED UP DATA PIPELINE
# ============================================================

AUTOTUNE = (
    tf.data.AUTOTUNE
)

train_data = (
    train_data.prefetch(
        buffer_size=AUTOTUNE
    )
)

validation_data = (
    validation_data.prefetch(
        buffer_size=AUTOTUNE
    )
)


# ============================================================
# BUILD CNN
# ============================================================

model = tf.keras.Sequential([

    # Normalize pixels from 0-255 to 0-1
    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # First feature detector
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Second feature detector
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Third feature detector
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Convert feature maps into a vector
    tf.keras.layers.Flatten(),

    # Classification layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Final prediction
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# SHOW MODEL
# ============================================================

print("\nModel architecture:")

model.summary()


# ============================================================
# TRAIN
# ============================================================

print(
    "\n" +
    "=" * 60
)

print(
    "STARTING TRAINING"
)

print(
    "=" * 60
)

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    f"\n[OK] Model saved to:"
)

print(
    MODEL_PATH
)