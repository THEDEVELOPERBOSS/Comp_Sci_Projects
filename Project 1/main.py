import tensorflow as tf

from coco_builder import build_coco_dataset


# ============================================================
# SETTINGS
# ============================================================
# ========================================================
# IMAGE SIZE
# ========================================================
#
# Every image is resized to 128 x 128 pixels before being
# given to the CNN.
#
# Neural networks need their input images to have the same
# dimensions.
#
# A larger image can contain more visual information,
# which may help the CNN recognize small details.
#
# However, larger images require more memory and take
# longer to process.
#
# 128 x 128 gives a good starting point because it is
# relatively fast to train.
#
# Experiment with 160 x 160 or 192 x 192 as desired
# and compare the accuracy and training time.
#
# ========================================================
IMAGE_SIZE = (128, 128)

BATCH_SIZE = 32
# ========================================================
# MAXIMUM NUMBER OF EPOCHS
# ========================================================
#
# An epoch is one complete pass through the training
# dataset.
#
# The model is allowed to train for up to 30 epochs.
#
# 30 is the MAXIMUM, not necessarily the actual number.
# Early stopping can stop training before reaching 30
# if validation accuracy stops improving.
#
# More epochs can give the CNN more opportunities to learn,
# but training for too long can cause overfitting.
#
# ========================================================
EPOCHS = 30

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


model = tf.keras.Sequential([

# ========================================================
# DATA AUGMENTATION
# ========================================================
#
# Data augmentation creates slightly different versions
# of training images.
#
# This helps prevent the CNN from simply memorizing the
# exact training images. Instead, it learns important
# features that help identify each object.
#
# For example, if the model sees a backpack from one angle,
# we want it to still recognize a backpack from a slightly
# different angle or position.
#
# RandomFlip:
# Randomly flips some training images horizontally.
#
# RandomRotation:
# Slightly rotates some training images.
# The value 0.1 allows a small amount of rotation.
#
# RandomZoom:
# Slightly zooms some training images in or out.
#
# These changes happen automatically during training.
# The original images stored on the computer are NOT changed.
#
# Data augmentation is only used on training data.
# Validation and test images remain unchanged.
#
# The goal is to make the model better at recognizing
# objects it has never seen before.
#
# ========================================================

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    ),

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
# ============================================================
# TRAINING CALLBACKS
# ============================================================

# ========================================================
# BEST MODEL CHECKPOINT
# ========================================================
#
# During training, the model's performance can improve
# and then get worse.
#
# We don't want to accidentally keep a worse version of
# the model.
#
# ModelCheckpoint automatically saves the model whenever
# validation accuracy reaches a new best value.
#
# monitor="val_accuracy":
# We use validation accuracy to decide which model is best.
#
# mode="max":
# A higher validation accuracy means a better model.
#
# save_best_only=True:
# Only save the model when it beats the previous best.
#
# MODEL_PATH:
# This is the file where the best model is saved.
#
# ========================================================
# Save the model whenever validation accuracy improves.
best_model_callback = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

# ========================================================
# EARLY STOPPING
# ========================================================
#
# Early stopping prevents the model from continuing to
# train when it stops getting better on validation data.
#
# The model could eventually start memorizing the training
# images instead of learning features that generalize to
# new images. This is called overfitting.
#
# monitor="val_accuracy":
# We watch the validation accuracy.
#
# mode="max":
# Higher validation accuracy is better.
#
# patience=3:
# Training is allowed to continue for 3 more epochs after
# validation accuracy stops improving.
#
# restore_best_weights=True:
# When training stops, TensorFlow restores the model weights
# from the epoch that had the best validation accuracy.
#
# This means we keep the best version of the model rather
# than simply keeping the final version.
#
# ========================================================
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\n" +
    "=" * 60
)

print("STARTING TRAINING")

print(
    "=" * 60
)

# ============================================================
# SAVE BEST MODEL
# ============================================================

model.save(MODEL_PATH)

print(
    f"\n[OK] Best model saved to:"
)

print(
    MODEL_PATH
)

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS
)

# ============================================================
# TEST MODEL
# ============================================================

print("\nLoading test data...")

test_data = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nTesting model...")

test_loss, test_accuracy = model.evaluate(test_data)

print(
    f"\nTest accuracy: "
    f"{test_accuracy * 100:.2f}%"
)