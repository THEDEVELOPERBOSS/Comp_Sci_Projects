# Image upload classifier

This Streamlit app lets you upload an image and classify it with the project's trained TensorFlow model. You can access it from another device on the same local network.

## Requirements

Use Python 3.12 as specified in `mise.toml`. From the `Project 1 and 2` directory, install the project's dependencies into your virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the interface for other devices

First, train the model using the existing training workflow so that `image_classifier.keras` exists in this directory. Then start Streamlit listening on your computer's network interfaces:

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m streamlit run image_classifier_app.py --server.address 0.0.0.0 --server.port 8501
```

**Linux (bash):**

```bash
.venv/bin/python -m streamlit run image_classifier_app.py --server.address 0.0.0.0 --server.port 8501
```

Keep the terminal open while using the app.

## Connect from another device

1. Connect both devices to the same trusted Wi-Fi or local network.
2. Find the IP address of the computer running Streamlit:
   - **Windows:** run `ipconfig` in PowerShell and find the IPv4 address for your active Wi-Fi or Ethernet adapter.
   - **Linux:** run `hostname -I` in a terminal and use the address for your local network.
3. On the other device, open a browser and visit `http://YOUR-COMPUTER-IP:8501`, replacing `YOUR-COMPUTER-IP` with that address. For example, if the computer's IP is `192.168.1.25`, visit `http://192.168.1.25:8501`.

If it does not load, check that the host computer's firewall allows Python/Streamlit connections on port 8501 on **private/local networks**. Avoid disabling the firewall entirely. Some guest Wi-Fi networks block communication between devices; use a trusted network that permits local-device connections.

The app accepts JPG, JPEG, PNG, and WEBP images and shows the predicted class, confidence, and probabilities for all eight classes. It loads the saved model only; it does not start dataset downloads or model training. It reads the expected input dimensions from the saved model and resizes uploaded images accordingly.

**Security note:** This setup is for access on a trusted local network. Do not port-forward port 8501 or expose the app directly to the public internet; the app has no authentication.


## Deploy from the main menu

You can also launch the web interface from the project's terminal menu:

1. Run the project's usual entry point, `main.py`.
2. Select **Deploy image classifier** from the main menu.
3. The program starts Streamlit, opens the page on the host computer, and prints the URL to type into a phone or other device on the same Wi-Fi network.
4. Keep the program/terminal and the Streamlit server running while people use the page.

The menu option checks that both `image_classifier_app.py` and `image_classifier.keras` exist before starting. It uses port `8501`. If another app already occupies that port, stop that app or change the port in `deploy_image_classifier()` in `main.py`.

If other devices cannot connect, allow Python/Streamlit through the host computer's firewall on private networks. Do not expose this unauthenticated app directly to the public internet.
