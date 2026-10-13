# ============================================================
# Imports 
# ============================================================
from pathlib import Path
import tensorflow as tf
import time
from pick import pick
from coco_builder import build_coco_dataset
from sync_dataset import pull 
from tabulate import tabulate
import os 
import json
import platform

# functionality for saving device specifc info 
def get_device_specs():
    gpus = tf.config.list_physical_devices("GPU")
    
    if gpus:
        gpu_name = gpus[0].name
    else:
        gpu_name = "No GPU detected"
        
    specs = {
        "operating_system": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "cpu_cores": os.cpu_count(),
        "gpu": gpu_name
    }
    
    return specs
# merges hardware records with what settings are currently in place
def create_run_record(test_accuracy):
    device_specs = get_device_specs()
    
    run_record = {
        "device": device_specs,
        "accuracy": test_accuracy, 
        "settings": {
            "epochs": EPOCHS,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "dropout": dropout_status
        }
    }
    
    return run_record

def save_training_run(run_record):
    """Save a training run without crashing if the history file is empty or corrupted."""

    runs = []

    # Load previous training runs if the file exists.
    if RUN_HISTORY_FILE.exists():
        try:
            with open(RUN_HISTORY_FILE, "r", encoding="utf-8") as file:
                content = file.read().strip()

            if content:
                runs = json.loads(content)
            else:
                raise json.JSONDecodeError(
                    "Training history file is empty", "", 0
                )

        except json.JSONDecodeError:
            # Preserve the damaged file before creating a new history.
            backup_path = RUN_HISTORY_FILE.with_name(
                "training_runs_corrupt_backup.json"
            )

            if not backup_path.exists():
                import shutil
                shutil.copy2(RUN_HISTORY_FILE, backup_path)

            print(
                "\nWarning: Training history was empty or invalid JSON."
            )
            print(f"A backup was saved to: {backup_path}")
            print("Starting a new training history.")

            runs = []

    # Add the latest training run.
    runs.append(run_record)

    # Save the updated history.
    with open(RUN_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(runs, file, indent=4)

    print("\nTraining run saved successfully.")

# ============================================================
# Function to clear terminal to make things cleaner
# ============================================================
def clear_terminal():
    print("\033[2J\033[H", end="")

# ============================================================
# Makes it so that the user has time to actually read/interact with what is happening on screen
# ============================================================
def give_time():
    input("Press enter to return...")
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

SETTINGS_FILE = PROJECT_DIR / "saved_settings.json"

RUN_HISTORY_FILE = PROJECT_DIR / "training_runs.json"


DATASET_DIR = PROJECT_DIR / "dataset"

# Original COCO dataset
TRAIN_DIR = DATASET_DIR / "classification" / "train"
VAL_DIR = DATASET_DIR / "classification" / "val"
TEST_DIR = DATASET_DIR / "classification" / "test"

# Open Images crops
CROPS_DIR = PROJECT_DIR / "open_images_crops"
OPEN_IMAGES_TRAIN_DIR = CROPS_DIR / "train"
OPEN_IMAGES_VAL_DIR = CROPS_DIR / "validation"
OPEN_IMAGES_TEST_DIR = CROPS_DIR / "test"

# Shared class order for the model
CLASS_NAMES = [
    "backpack",
    "bottle",
    "car",
    "couch",
    "laptop",
    "mobile_phone",
    "table",
    "truck",
]

# Original dataset calls mobile_phone "phone"
ORIGINAL_CLASS_NAMES = [
    "backpack",
    "bottle",
    "car",
    "couch",
    "laptop",
    "phone",
    "table",
    "truck",
]

MODEL_PATH = PROJECT_DIR / "image_classifier.keras"


def load_saved_settings():

    if not SETTINGS_FILE.exists():
        return {}

    with open(SETTINGS_FILE, "r") as file:
        return json.load(file)

def save_saved_settings(saved_settings):

    with open(SETTINGS_FILE, "w") as file:
        json.dump(saved_settings, file, indent=4)

# ============================================================
# SETTINGS
# ============================================================
PATIENCE = 3
# ========================================================
# IMAGE SIZE
# ========================================================
#
# Every image is resized to the image size in pixels before being
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
IMAGE_SIZE = (128, 128)
# ========================================================
# BATCH_SIZE
#
# Makes predictions on # of images
# Compares predicitons to correct answers
# Calculates the error and updates the weights
# Moves on to next # of images
# 
# ========================================================
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
# Dropout on/off
# ============================================================
dropout_status = "INACTIVE"
# ============================================================
# Load saved settings
# ============================================================
saved_settings = load_saved_settings()
# ============================================================
# BUILD / CHECK DATASETS
# ============================================================
pull() # uses sync_dataset.py function 

build_coco_dataset()
# ============================================================
# Save current settings function 
# ============================================================
def save_current_settings():

    global saved_settings

    name = input("\nWhat would you like to name these settings? ")

    saved_settings[name] = {
        "epochs": EPOCHS,
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "dropout": dropout_status
    }

    save_saved_settings(saved_settings)

    print(f"\n'{name}' has been saved.")

    give_time()


def load_combined_split(original_dir, open_images_dir, split_name, shuffle=False):
    """Load the original images and Open Images crops together."""

    print(f"\nLoading combined {split_name} data...")

    # Load the original COCO dataset.
    original_data = tf.keras.utils.image_dataset_from_directory(
        original_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        class_names=ORIGINAL_CLASS_NAMES,
    )

    # Load the Open Images crops.
    open_images_data = tf.keras.utils.image_dataset_from_directory(
        open_images_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        class_names=CLASS_NAMES,
    )

    # Combine both datasets.
    combined_data = original_data.unbatch().concatenate(
        open_images_data.unbatch()
    )

    # Shuffle training images, but keep validation and test ordered.
    if shuffle:
        combined_data = combined_data.shuffle(
            buffer_size=512,
            seed=42,
            reshuffle_each_iteration=True,
        )

    return combined_data.batch(BATCH_SIZE)

def load_training_data():
    return load_combined_split(
        TRAIN_DIR,
        OPEN_IMAGES_TRAIN_DIR,
        "training",
        shuffle=True,
    )


def load_validation_data():
    return load_combined_split(
        VAL_DIR,
        OPEN_IMAGES_VAL_DIR,
        "validation",
        shuffle=False,
    )

# ============================================================
# GET CLASS NAMES AND SPEED UP DATA PIPELINE(PREPARE DATA)
# ============================================================
def prepare_data(train_data, validation_data):
    class_names = CLASS_NAMES
    NUM_CLASSES = len(class_names)

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
    # speed up data pipeline
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
    return train_data, validation_data, NUM_CLASSES
def dropout():
    if dropout_status == "ACTIVE":
        return tf.keras.layers.Dropout(0.5)

    # Return a layer that does nothing when dropout is inactive.
    return tf.keras.layers.Activation("linear")
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
         


def build_model(NUM_CLASSES):
    # Randomly change training images to help prevent overfitting.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Load MobileNetV2's pretrained visual features.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )

    # Train only our new classification layers initially.
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    x = data_augmentation(inputs)

    # Convert pixels from 0–255 to -1–1.
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    )(x)

    # Extract features without updating MobileNetV2's weights.
    x = base_model(x, training=False)

    # Reduce the feature maps to a compact feature vector.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Add regularization to help reduce overfitting.
    x = tf.keras.layers.Dropout(0.2)(x)

    # Produce a probability for each class.
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="class_predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model

def compile(model):
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
    return model
def show_model(model):
    # ============================================================
    # SHOW MODEL
    # ============================================================

    print("\nModel architecture:")

    model.summary()
    return model

def training_callbacks():
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
    return best_model_callback
def early_stopping():
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
    # Verbose is how much information it prints while it runs
    # verbose=0: Prints nothing. Silent 
    #
    # verbose=1: prints info about what is happening. 
    # Ex:For EarlyStopping a message will appear that training has stopped
    #
    # verbose=2: More detailed output in some keras functions
    # Ex: for model.fit() it usually prints one line per epoch rather
    # than the animated progress bar
    # ========================================================
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    return early_stopping_callback

def train(model, train_data, validation_data, best_model_callback, early_stopping_callback):    
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
    return model, train_data, validation_data, best_model_callback, early_stopping_callback

def test_model(model):
    """Test using both datasets."""

    test_data = load_combined_split(
        TEST_DIR,
        OPEN_IMAGES_TEST_DIR,
        "testing",
        shuffle=False,
    )

    print("\nTesting model...")

    results = model.evaluate(test_data, return_dict=True)
    test_accuracy = results["accuracy"]

    print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

    return test_accuracy

def choose_image_size():

    # ========================================================
    # IMAGE SIZE OPTIONS
    # ========================================================
    #
    # Each tuple contains:
    #
    # [0] = Name displayed to the user
    # [1] = Actual image size used by TensorFlow
    # [2] = Training speed
    # [3] = Amount of detail
    # [4] = Memory usage
    # [5] = What the size is good for
    #
    image_sizes = [
        ("64x64",   (64, 64),   "⚡ Very fast",      "Low",            "Very low",      "Testing/debugging"),
        ("96x96",   (96, 96),   "⚡ Fast",           "Low–medium",     "Low",           "Basic experiments"),
        ("128x128", (128, 128), "🟢 Fast",           "Medium",         "Low–medium",    "Good starting point"),
        ("160x160", (160, 160), "🟢 Medium",         "Medium–high",    "Medium",        "More detailed classification"),
        ("192x192", (192, 192), "🟡 Slower",         "High",           "Medium",        "Detailed objects"),
        ("224x224", (224, 224), "🟡 Slower",         "High",           "Medium–high",   "Common CNN size"),
        ("256x256", (256, 256), "🟠 Slow",           "Very high",      "High",          "More demanding classification"),
        ("320x320", (320, 320), "🔴 Very slow",      "Very high",      "High",          "Fine details"),
        ("384x384", (384, 384), "🔴 Extremely slow", "Extremely high", "Very high",      "INSANE")
    ]


    # ========================================================
    # CREATE THE TABLE
    # ========================================================
    #
    # tabulate turns our image-size information into a
    # readable table.
    #
    table = tabulate(
        image_sizes,
        headers=[
            "Image Size",
            "TensorFlow Value",
            "Training Speed",
            "Detail",
            "Memory Usage",
            "Good For"
        ],
        tablefmt="rounded_outline"
    )


    # ========================================================
    # CREATE THE PICK OPTIONS
    # ========================================================
    #
    # pick() needs a simple list of choices.
    #
    # We use only the image-size names for the choices.
    #
    choices = []

    for row in image_sizes:

        choices.append(row[0])


    # ========================================================
    # CREATE THE PICK TITLE
    # ========================================================
    #
    # Instead of using print() for the table, we put the
    # table directly into pick().
    #
    # This is important because pick() controls the terminal
    # screen and may clear anything printed before it.
    #
    title = (
        "Select an image size:\n\n"
        + table
        + "\n\n"
        + "Use ↑/↓ to move and ENTER to select:"
    )


    # ========================================================
    # LET THE USER SELECT AN IMAGE SIZE
    # ========================================================
    #
    # pick() now displays:
    #
    # 1. The full table
    # 2. The selectable image-size options
    #
    # The user can move through the options using the
    # arrow keys.
    #
    selected, index = pick(
        choices,
        title,
        indicator="=>",
        default_index=2
    )


    # ========================================================
    # RETURN THE ACTUAL IMAGE SIZE
    # ========================================================
    #
    # index tells us which image-size row the user selected.
    #
    # [1] gets the actual (width, height) value that
    # TensorFlow uses.
    #
    # Example:
    #
    # image_sizes[4]
    #
    # gives:
    #
    # ("192x192", (192, 192), ...)
    #
    # image_sizes[4][1]
    #
    # gives:
    #
    # (192, 192)
    #
    return image_sizes[index][1]
# ============================================================
# SETTINGS FUNCTIONS
# ============================================================
def change_epochs():
    clear_terminal()
    global EPOCHS

    print("How many epochs do you want?")
    EPOCHS = int(input())

    print(f"The model will run for {EPOCHS} epochs")
    
    give_time()
    
def change_image_size():
    clear_terminal()
    global IMAGE_SIZE
    
    IMAGE_SIZE = choose_image_size()

    give_time()
    
def change_batch_size():
    clear_terminal()
    global BATCH_SIZE

    print("What would you like to change batch size to?")
    BATCH_SIZE = int(input())
    
    print(f"Batch size is now set to {BATCH_SIZE}")

    give_time()
    
def change_patience():
    clear_terminal()
    global PATIENCE
    print("What would you like to change patience to?")
    PATIENCE = int(input())
    
    print(f"Patience is now set to {PATIENCE}")

    give_time()
def change_dropout():
    clear_terminal()
    global dropout_status
    if dropout_status == "ACTIVE":
        dropout_status = "INACTIVE"
        print(f"\nDropout is now set to: {dropout_status}")
        
        give_time()
    else: 
        dropout_status = "ACTIVE"
        print(f"\nDropout is now set to: {dropout_status}")
        
        give_time()
def see_current_settings():
    clear_terminal()
    global dropout_status
    print("\nHere are your current settings:")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Dropout: {dropout_status}")
    print(f"Patience: {PATIENCE}")
    
    give_time()
# ============================================================
# SAVED SETTINGS FUNCTIONS
# ============================================================
def create_new_profile():

    global saved_settings

    step = 0

    profile = {
        "name": "",
        "epochs": None,
        "image_size": None,
        "batch_size": None,
        "patience": None, 
        "dropout": None
    }
    
    clear_terminal()
    while step >= 0 and step <= 5:
        
        print("Create New Profile\n")
        
        # Profile name
        if step == 0:
            name = input("What would you like to name this profile? ")
            profile["name"] = name
            step += 1 
        # Epochs
        elif step == 1:
            
            clear_terminal()
            print("Create New Profile\n")
            print(f"How many epochs would you like?\n")
            print("Type b to go back")
            print("\nDefault:")
            if profile["epochs"] is None:
                print(30)
            else:
                print(profile["epochs"])
            epochs = input("> ")
            if epochs.lower() == "b":
                step -= 1
                continue
            
            profile["epochs"] = int(epochs)
            step += 1 
        
        # Image size
        elif step == 2:
            clear_terminal()
            print("Create New Profile\n")
            print("Choose an image size:\n")
            
            image_size = choose_image_size()

            profile["image_size"] = image_size
            step += 1
        # Batch size
        elif step == 3:
            clear_terminal()
            print("Create New Profile\n")
            print("What batch size would you like\n")
            print("Type b to go back")
            if profile["batch_size"] is None:
                print(32)
            else:
                print(profile["batch_size"])
            batch_size = input("> ")
            if batch_size.lower() == "b":
                step -= 1 
                continue
            
            profile["batch_size"] = int(batch_size)
            step += 1 
        # Patience
        elif step == 4:
            clear_terminal()
            print("Create New Profile\n")
            print("What patience would you like?\n")
            print("Type b to go back)")
            print("\nDefault:")
            if profile["patience"] is None:
                print(3)
            else:
                print(profile["patience"])
            patience = input("> ")
            if patience.lower() == "b":
                step -= 1 
                continue
            
            profile["patience"] = int(patience)
            step += 1 

        # Dropout
        elif step == 5:
            clear_terminal()
            print("Create New Profile\n")
            print("Would you like dropout to be active?\n")
            
            dropout_options = ["ACTIVE", "INACTIVE"]
            
            dropout, index = pick(
                dropout_options, 
                "Select dropout setting:",
                indicator = "=>",
                default_index = 1
            )
            
            profile["dropout"] = dropout
            step += 1 
        # Create the profile
        saved_settings[profile["name"]] = {
            "epochs": profile["epochs"],
            "image_size": profile["image_size"],
            "batch_size": profile["batch_size"],
            "patience": profile["patience"],
            "dropout": profile["dropout"]
        }

        # Save it to JSON
        save_saved_settings(saved_settings)

        clear_terminal()

        print(f"Profile '{name}' has been created successfully!")

        give_time()
def saved_settings_menu():
    
    global EPOCHS
    global IMAGE_SIZE
    global BATCH_SIZE
    global PATIENCE
    global dropout_status

    while True:

        clear_terminal()

        print("Saved Settings:\n")

        if not saved_settings:
            print("No saved settings yet.")
        else:
            for name in saved_settings:
                print(f"- {name}")

        print("\nWhat would you like to do?")

        options = ["Load saved settings", "Create new saved settings", "Back"]

        choice, index = pick(options, "Select an option:", indicator="=>", default_index=0)

        if choice == "Create new saved settings":

            clear_terminal()
            
            print("\nWhat would you like to do?")
            
            options=["Save settings currently in place", "Make a new profile", "Back"]
            
            choice, index = pick(options, "Select an option:", indicator="=>", default_index=0)
            if choice == "Save settings currently in place":
                save_current_settings()
            elif choice == "Make a new profile":
                create_new_profile()
            elif choice == "Back":
                return

        elif choice == "Load saved settings":

            if not saved_settings:
                clear_terminal()
                print("There are no saved settings to load.")
                give_time()
                continue
            
            names = list(saved_settings.keys())

            selected_name, index = pick(
                names,
                "Choose saved settings:",
                indicator="=>",
                default_index=0
            )

            settings = saved_settings[selected_name]

            EPOCHS = settings["epochs"]
            IMAGE_SIZE = tuple(settings["image_size"])
            BATCH_SIZE = settings["batch_size"]
            PATIENCE = settings["patience"]
            dropout_status = settings["dropout"]

            clear_terminal()

            print(f"Loaded '{selected_name}'.")

            give_time()

        elif choice == "Back":

            return
# ============================================================
# LEARN FUNCTIONS
# ============================================================
def learn_epochs():
    clear_terminal()
    print("\nEpochs are the amount of times it will train before the model gets tested.\n")
    print(f"The model is also designed to stop training if no improvement is made in {PATIENCE} runs")
    
    give_time()
    
def learn_image_size():
    clear_terminal()
    print("\nEvery image is resized to the image size in pixels before being given to the CNN.")
    print("Neural networks need their input images to have the same dimensions.")
    print("A larger image size can contain more visual information, \nwhich may help the CNN recognize small details.")
    print("However, larger images require more memory and take longer to process.")
    
    give_time()

def learn_batch_size():
    clear_terminal()
    print("\nBatch size controls how many images are processed by the CNN at one time.")
    print("A larger batch size can make training faster, \nbut it requires more memory.")
    print("A smaller batch size uses less memory, \nbut training may take longer.")
    print("The batch size can also affect how the model learns from the training data.")
    
    give_time()
    
def learn_patience():
    clear_terminal()
    print("\nPatience controls how many training epochs the model waits for improvement.")
    print("If the validation accuracy stops improving, \nthe model will continue training for the number of epochs set by patience.")
    print("If there is still no improvement after that, training will stop early.")
    print("This can save time and help prevent the model from training longer than necessary.")
    
    give_time()
    
# ============================================================
# MENUS
# ============================================================
settings_menu = {
    "Epochs": change_epochs,
    "Image Size": change_image_size,
    "Batch Size": change_batch_size,
    "Patience": change_patience,
    "Dropout": change_dropout
}
learn_menu = {
    "Epochs": learn_epochs,
    "Image Size": learn_image_size,
    "Batch Size": learn_batch_size,
    "Patience": learn_patience
}
def main():
    while True:
        clear_terminal()
        # First UI. Maybe add a way to reset to defaults.
        user_input = (
            "Would you like to: \n"
            "Run with the defaults \n"
            "Change settings \n"
            "Learn \n"
            "Choose a set of saved settings\n"
            "See current settings"
        )
        options = [
            "Defaults",
            "Change Settings",
            "Learn",
            "Saved settings",
            "See current settings",
        ]
        option, index = pick(options, user_input, indicator="=>", default_index=0)

        # Defaults
        if option == "Defaults":
            clear_terminal()
            print("Defaults selected. Beginning training run")
            train_data = load_training_data()
            validation_data = load_validation_data()
            train_data, validation_data, NUM_CLASSES = prepare_data(train_data, validation_data)
            model = build_model(NUM_CLASSES)
            model = compile(model)
            show_model(model)
            best_model_callback = training_callbacks()
            early_stopping_callback = early_stopping()
            train(
                model,
                train_data,
                validation_data,
                best_model_callback,
                early_stopping_callback
            )

            # Load the best saved model before testing.
            best_model = tf.keras.models.load_model(MODEL_PATH)

            test_accuracy = test_model(best_model)

                        
            run_record = create_run_record(test_accuracy)
            
            save_training_run(run_record)

        # Change settings
        elif option == "Change Settings":
            clear_terminal()
            user_input = "What would you like to change? "
            options = ["Epochs", "Image Size", "Batch Size", "Dropout","Patience"]
            option, index = pick(options, user_input, indicator="=>", default_index=0)

            settings_menu[option]()

        elif option == "See current settings":
            see_current_settings()

        elif option == "Learn":
            clear_terminal()
            user_input = "What would you like to learn about? "
            options = ["Epochs", "Image Size", "Batch Size", "Patience"]
            option, index = pick(options, user_input, indicator="=>", default_index=0)
            
            learn_menu[option]()
        elif option == "Saved settings":
            saved_settings_menu()
if __name__ == "__main__":
    main()