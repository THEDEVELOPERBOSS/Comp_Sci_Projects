from pathlib import Path

import requests
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

# This file's folder.
# This makes the downloader portable between computers.
PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"


# ============================================================
# FILES TO DOWNLOAD
# ============================================================

FILES = {
    "annotations_trainval2017.zip": {
        "url": (
            "http://images.cocodataset.org/annotations/"
            "annotations_trainval2017.zip"
        ),

        "destination": (
            ANNOTATIONS_DIR /
            "annotations_trainval2017.zip"
        ),
    },
}


# ============================================================
# RESUMABLE DOWNLOAD
# ============================================================

def download_with_resume(url, destination):
    """
    Download a file with resume support.

    Complete file:
        filename.zip

    Partial file:
        filename.zip.part

    If the download is interrupted, run main.py again
    and it will attempt to continue from where it stopped.
    """

    destination = Path(destination)

    partial_path = Path(
        str(destination) + ".part"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECK FOR COMPLETE FILE
    # --------------------------------------------------------

    if destination.exists():

        print(
            f"[OK] {destination.name} "
            f"already exists."
        )

        return True

    # --------------------------------------------------------
    # CHECK FOR PARTIAL FILE
    # --------------------------------------------------------

    existing_size = 0

    if partial_path.exists():

        existing_size = (
            partial_path.stat().st_size
        )

        print(
            f"[RESUME] {destination.name}"
        )

        print(
            f"Existing download: "
            f"{existing_size / (1024 * 1024):.1f} MB"
        )

    else:

        print(
            f"[DOWNLOAD] "
            f"{destination.name}"
        )

    # --------------------------------------------------------
    # CREATE RANGE HEADER
    # --------------------------------------------------------

    headers = {}

    if existing_size > 0:

        headers["Range"] = (
            f"bytes={existing_size}-"
        )

    try:

        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=60
        ) as response:

            # ------------------------------------------------
            # CHECK RESPONSE
            # ------------------------------------------------

            response.raise_for_status()

            # ------------------------------------------------
            # SERVER ACCEPTED RESUME
            # ------------------------------------------------

            if (
                existing_size > 0
                and response.status_code == 206
            ):

                mode = "ab"

                remaining_size = int(
                    response.headers.get(
                        "content-length",
                        0
                    )
                )

                total_size = (
                    existing_size +
                    remaining_size
                )

                initial_progress = (
                    existing_size
                )

                print(
                    "[OK] Resuming download..."
                )

            # ------------------------------------------------
            # START NEW DOWNLOAD
            # ------------------------------------------------

            else:

                # If a partial file exists but the server
                # ignored our Range request, restart safely.

                if existing_size > 0:

                    print(
                        "[WARNING] Server did not "
                        "accept resume request."
                    )

                    print(
                        "Restarting this file..."
                    )

                mode = "wb"

                total_size = int(
                    response.headers.get(
                        "content-length",
                        0
                    )
                )

                initial_progress = 0

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            with open(
                partial_path,
                mode
            ) as file:

                with tqdm(
                    total=total_size,
                    initial=initial_progress,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name
                ) as progress:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if chunk:

                            file.write(chunk)

                            progress.update(
                                len(chunk)
                            )

        # ----------------------------------------------------
        # DOWNLOAD COMPLETE
        # ----------------------------------------------------

        partial_path.replace(
            destination
        )

        print(
            f"[COMPLETE] "
            f"{destination.name}"
        )

        return True

    except KeyboardInterrupt:

        print(
            "\n[PAUSED] Download interrupted."
        )

        print(
            "Progress was saved."
        )

        print(
            "Run main.py again later "
            "to continue."
        )

        return False

    except Exception as error:

        print(
            f"\n[ERROR] "
            f"{destination.name}"
        )

        print(
            f"Reason: {error}"
        )

        print(
            "Partial download was kept."
        )

        return False


# ============================================================
# CHECK ALL REQUIRED FILES
# ============================================================

def ensure_dataset_files():
    """
    Check every required file.

    - Complete files are skipped.
    - Missing files are downloaded.
    - Partial files are resumed.
    """

    print("\n" + "=" * 60)

    print(
        "CHECKING DATASET FILES"
    )

    print("=" * 60 + "\n")

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_complete = True

    for (
        file_name,
        file_info
    ) in FILES.items():

        success = (
            download_with_resume(
                url=file_info["url"],
                destination=file_info[
                    "destination"
                ]
            )
        )

        if not success:

            all_complete = False

    print("\n" + "=" * 60)

    if all_complete:

        print(
            "ALL REQUIRED FILES ARE READY."
        )

    else:

        print(
            "SOME FILES ARE NOT COMPLETE."
        )

        print(
            "Run the program again "
            "to continue downloading."
        )

    print("=" * 60 + "\n")

    return all_complete