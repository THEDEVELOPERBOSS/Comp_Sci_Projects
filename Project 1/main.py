from pathlib import Path
import tensorflow as tf
import time

from coco_builder import build_coco_dataset


# ============================================================
# PROJECT PATHS
# ============================================================
#
# __file__ is the location of main.py.
# Using it means the program finds the dataset relative to
# the project, regardless of which folder PowerShell is in.
#
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "classification" / "train"

VAL_DIR = DATASET_DIR / "classification" / "val"

TEST_DIR = DATASET_DIR / "classification" / "test"

MODEL_PATH = PROJECT_DIR / "image_classifier.keras"


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
# (128, 128) → faster, less detail
# (160, 160) → good middle ground
# (192, 192) → more detail, slower training
#
# Experiment with 160 x 160 or 192 x 192 as desired
# and compare the accuracy and training time.
#
# ========================================================
IMAGE_SIZE = (160, 160) # Change this to change image size

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
        128, # number of filters
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

# Convert feature maps into a vector
tf.keras.layers.Flatten(),

# Classification layer
tf.keras.layers.Dense(
    128, # number of neurons
    activation="relu"
),

# ========================================================
# DROPOUT
# ========================================================
#
# Dropout helps prevent overfitting.
#
# During training, it temporarily disables a random
# percentage of neurons in the previous layer.
#
# With 0.5, approximately 50% of those neurons are
# temporarily ignored during each training step.
#
# This forces the network to learn using multiple useful
# features instead of depending too heavily on specific
# neurons.
#
# Dropout is only active during training. When the model
# is tested, all neurons are used normally.
#
# Currently removed due to harming performance as of removing it
#
# ========================================================

# tf.keras.layers.Dropout(
#    0.5
#),

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

# Starts a stopwatch to keep track of time 

start_time = time.time()

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[
        best_model_callback,
        early_stopping_callback
    ]
)

# Stops stopwatch
end_time = time.time()

# Calculate how training took
training_time = end_time - start_time

minutes = int(training_time // 60)
seconds = int(training_time % 60)

print(
    f"\nTraining time: {minutes} minutes {seconds} seconds"
)

print("\n[OK] Best model saved to:")
print(MODEL_PATH)
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
def main():
    print(f"Best run: ") 
    print(f"Here are you current settings:")
    user_input = input("")