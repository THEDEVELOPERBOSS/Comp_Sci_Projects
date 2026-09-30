Use Python 3.12 for this project, as specified in `mise.toml`. TensorFlow is
not available for the Python 3.14 interpreter previously used to run it.

From this project directory, create an isolated environment and install the
requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the project with:

```powershell
.\.venv\Scripts\python.exe main.py
```