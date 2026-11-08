"""Download the Rossmann train/store CSVs into data/raw.

Source: Rossmann Store Sales (Dirk Rossmann GmbH), via Kaggle
https://www.kaggle.com/c/rossmann-store-sales/data
The two CSVs are mirrored as a zip on Google Drive so reviewers need no Kaggle account.

Usage: uv run python scripts/download_data.py [--force]

Steps:
    1. Create data/raw if it does not exist.
    2. Skip everything if train.csv and store.csv are already there (unless --force).
    3. Download the zip from Google Drive into a temporary folder.
    4. Check that the download is really a zip (Drive returns a web page if sharing is off).
    5. Unzip the CSVs into data/raw.
    6. Verify both files exist and print their sizes.
"""

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

FILE_ID = "1J-ayJE3WybFkGRZsNQJnFpj9BEUlg_MH"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = ["train.csv", "store.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.force and all((RAW_DIR / name).exists() for name in EXPECTED):
        print(f"Data already in {RAW_DIR}. Use --force to re-download.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "rossmann-data.zip"
        print("Downloading from Google Drive...")
        urllib.request.urlretrieve(URL, zip_path)
        if not zipfile.is_zipfile(zip_path):
            sys.exit(
                "Download failed: Drive did not return a zip file. "
                "Check that sharing is 'Anyone with the link', or get the data from the Kaggle link in the README."
            )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)

    for name in EXPECTED:
        path = RAW_DIR / name
        if not path.exists():
            sys.exit(f"Expected file missing after unzip: {name}")
        print(f"{name}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
