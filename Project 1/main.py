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

MODEL_PATH = "image_classifier.keras"


# ============================================================
# BUILD / CHECK DATASET
# ============================================================

build_coco_dataset()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

validation_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# GET CLASS NAMES
# ============================================================

class_names = train_data.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

NUM_CLASSES = len(class_names)

print(f"\nNumber of classes: {NUM_CLASSES}")


# ============================================================
# IMPROVE DATA PIPELINE SPEED
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

validation_data = validation_data.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# BUILD NEURAL NETWORK
# ============================================================

model = tf.keras.Sequential([ # Sequential takes the output of one layer and feed it directly into the next layer

    # Convert pixel values from 0-255 to 0-1
    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # First convolution layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3), # means it looks at 3 x 3 pixel area at a time to crossrefernce with what it has already learned
        activation="relu" # Rectified Linear Unit
    ),

    tf.keras.layers.MaxPooling2D(),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Third convolution layer
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Turn the feature maps into a vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output layer
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# SHOW MODEL
# ============================================================

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

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
    f"\nModel saved to: "
    f"{MODEL_PATH}"
)