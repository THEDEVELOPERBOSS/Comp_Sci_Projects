# Image upload classifier

This Streamlit app lets you upload an image and classify it with the project's trained TensorFlow model.

## Requirements

Use Python 3.12 as specified in `mise.toml`. From the `Project 1 and 2` directory, install the project's dependencies into your virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the interface

First, train the model using the existing training workflow so that `image_classifier.keras` exists in this directory. Then start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run image_classifier_app.py
```

Streamlit opens the interface in your browser. Choose a JPG, JPEG, PNG, or WEBP image, then select **Classify image**. The app shows the predicted class, confidence, and probabilities for all eight classes.

The app loads the saved model only; it does not start dataset downloads or model training. It reads the expected input dimensions from the saved model and resizes the uploaded image accordingly.
