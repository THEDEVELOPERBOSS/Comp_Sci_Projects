
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Find files beside this script.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "image_classifier.keras"
DATASET_DIR = PROJECT_DIR / "dataset" / "classification" / "train"

IMAGE_SIZE = (160, 160)

st.set_page_config(
    page_title="Image Recognition",
    page_icon="📷",
)

st.title("My Image Recognition Software")
st.write("Take a picture and let my AI classify it.")

# Load the model once and reuse it.
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Match prediction indexes to training class names.
@st.cache_data
def get_class_names():
    return sorted(
        folder.name for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

if not DATASET_DIR.exists():
    st.error(f"Training folders not found: {DATASET_DIR}")
    st.stop()

try:
    model = load_model()
    class_names = get_class_names()
    st.write("Model output count:", model.output_shape[-1])
    st.write("Training folder count:", len(class_names))
    st.write("Training folder names:", class_names)
except Exception as error:
    st.error(f"Could not load the model or class names: {error}")
    st.stop()

photo = st.camera_input("Take a picture")

if photo is not None:
    image = Image.open(photo).convert("RGB")
    st.image(image, caption="Your photo", use_container_width=True)

    if st.button("Classify Image", type="primary"):
        # Resize and prepare the image for the model.
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Analyzing image..."):
            try:
                predictions = model.predict(
                    image_array, verbose=0
                )[0]

                if len(predictions) != len(class_names):
                    st.error(
                        "The number of model outputs does not "
                        "match the number of training classes."
                    )
                    st.stop()

                best_index = int(np.argmax(predictions))

                st.subheader("Prediction")
                st.success(class_names[best_index])
                st.write(
                    f"Model score: "
                    f"{predictions[best_index] * 100:.2f}%"
                )

                st.subheader("All class scores")

                for index in np.argsort(predictions)[::-1]:
                    st.write(
                        f"{class_names[index]}: "
                        f"{predictions[index] * 100:.2f}%"
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
