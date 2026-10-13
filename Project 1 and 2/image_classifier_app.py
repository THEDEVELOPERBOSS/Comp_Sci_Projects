"""Streamlit interface for classifying a user-uploaded image.

Run from this directory with:
    streamlit run image_classifier_app.py

The trained model must exist at image_classifier.keras.
"""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "image_classifier.keras"

# Keep this order aligned with CLASS_NAMES in main.py and the model's output.
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

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")
st.title("Image Classifier")
st.write("Upload an image and let the trained model predict its class.")

if not MODEL_PATH.is_file():
    st.error(
        "The trained model was not found. Train the model first so that "
        "`image_classifier.keras` exists in this project directory."
    )
    st.stop()


@st.cache_resource
def load_model(model_path: str):
    """Load the trained model once and reuse it between Streamlit reruns."""
    return tf.keras.models.load_model(model_path)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, and WEBP.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be opened as a supported image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify image", type="primary"):
        try:
            model = load_model(str(MODEL_PATH))
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            height, width = input_shape[1], input_shape[2]
            if height is None or width is None:
                st.error("The model does not define a fixed image size.")
                st.stop()

            # Match the dimensions used to train this saved model.
            resized_image = image.resize((width, height))
            image_array = np.asarray(resized_image, dtype=np.float32)
            image_batch = np.expand_dims(image_array, axis=0)

            with st.spinner("Classifying image..."):
                predictions = model.predict(image_batch, verbose=0)[0]

            if len(predictions) != len(CLASS_NAMES):
                st.error(
                    "The model output does not match the eight configured "
                    "class names. Check CLASS_NAMES against the model training code."
                )
                st.stop()

            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index])

            st.subheader("Prediction")
            st.success(f"Class: {CLASS_NAMES[predicted_index].replace('_', ' ').title()}")
            st.metric("Confidence", f"{confidence:.1%}")

            st.subheader("Class probabilities")
            probability_table = {
                "Class": [name.replace("_", " ").title() for name in CLASS_NAMES],
                "Probability": [float(value) for value in predictions],
            }
            st.dataframe(
                probability_table,
                hide_index=True,
                use_container_width=True,
            )
        except (OSError, ValueError, TypeError) as error:
            st.error(f"Could not classify this image: {error}")
