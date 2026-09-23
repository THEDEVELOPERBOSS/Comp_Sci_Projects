import subprocess
import sys
from pathlib import Path

REPO = "THEDEVELOPERBOSS/Image_classification"
DATASET = Path(__file__).parent / "dataset"
# Push command: 
# python sync_dataset.py push
# Pull command
# python sync_dataset.py pull
def run(command):
    print("\n>", " ".join(command))
    result = subprocess.run(command)

    if result.returncode != 0:
        print("\n❌ Command failed.")
        sys.exit(result.returncode)


def find_hf():
    # Try hf from PATH
    try:
        result = subprocess.run(
            ["hf", "--version"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return "hf"

    except FileNotFoundError:
        pass

    # Look next to the Python executable
    scripts_folder = Path(sys.executable).parent
    hf_exe = scripts_folder / "hf.exe"

    if hf_exe.exists():
        return str(hf_exe)

    # Windows user-install location
    user_scripts = (
        Path.home()
        / "AppData"
        / "Local"
        / "Packages"
        / "PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0"
        / "LocalCache"
        / "local-packages"
        / "Python313"
        / "Scripts"
        / "hf.exe"
    )

    if user_scripts.exists():
        return str(user_scripts)

    print("❌ Hugging Face CLI (hf.exe) was not found.")
    print("Install it with:")
    print("pip install -U huggingface_hub")
    sys.exit(1)
    """
    Find the Hugging Face CLI regardless of where
    Python/pip installed it.
    """

    # Try the normal PATH first
    try:
        result = subprocess.run(
            ["hf", "--version"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return "hf"

    except FileNotFoundError:
        pass

    # Try the same Python environment that is running this script
    scripts_folder = Path(sys.executable).parent

    hf_exe = scripts_folder / "hf.exe"

    if hf_exe.exists():
        return str(hf_exe)

    # Try Python's module execution as a final option
    return [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli"]


def push():
    if not DATASET.exists():
        print(f"❌ Dataset folder not found:")
        print(DATASET)
        return

    hf = find_hf()

    print("⬆️  Syncing local dataset → Hugging Face...")

    if isinstance(hf, list):
        command = hf + [
            "upload",
            REPO,
            str(DATASET),
            ".",
            "--repo-type=dataset",
            "--delete=*"
        ]
    else:
        command = [
            hf,
            "upload",
            REPO,
            str(DATASET),
            ".",
            "--repo-type=dataset",
            "--delete=*"
        ]

    run(command)

    print("\n✅ Dataset pushed successfully.")


def pull():
    DATASET.mkdir(parents=True, exist_ok=True)

    hf = find_hf()

    print("⬇️ Syncing Hugging Face → local dataset...")

    if isinstance(hf, list):
        command = hf + [
            "download",
            REPO,
            "--repo-type=dataset",
            "--local-dir",
            str(DATASET)
        ]
    else:
        command = [
            hf,
            "download",
            REPO,
            "--repo-type=dataset",
            "--local-dir",
            str(DATASET)
        ]

    run(command)

    print("\n✅ Dataset pulled successfully.")


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python sync_dataset.py push")
        print("  python sync_dataset.py pull")
        return

    command = sys.argv[1].lower()

    if command == "push":
        push()
    elif command == "pull":
        pull()
    else:
        print("❌ Unknown command.")
        print("Use:")
        print("  python sync_dataset.py push")
        print("  python sync_dataset.py pull")


if __name__ == "__main__":
    main()