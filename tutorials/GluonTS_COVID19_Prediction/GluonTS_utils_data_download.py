"""
Simple data downloader for COVID-19 datasets.

Downloads data from Google Drive if not present locally.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.GluonTS_utils_data_download as ttgcpgudd
"""

from pathlib import Path
import urllib.request


def download_file_from_google_drive(file_id, destination):
    """
    Download file from Google Drive.

    Args:
        file_id: Google Drive file ID
        destination: Local path to save file

    Returns:
        bool: True if successful, False otherwise
    """

    def get_confirm_token(response):
        for key, value in response.headers.items():
            if key.lower() == "set-cookie":
                if "download_warning" in value:
                    return value.split("download_warning=")[1].split(";")[0]
        return None

    print(f"Downloading {destination.name}... ", end="", flush=True)

    try:
        URL = "https://drive.google.com/uc?export=download"

        session = urllib.request.build_opener()

        # Initial request
        response = session.open(f"{URL}&id={file_id}")
        token = get_confirm_token(response)

        # If large file, need confirmation token
        if token:
            response = session.open(f"{URL}&id={file_id}&confirm={token}")

        # Save file
        with open(destination, "wb") as f:
            f.write(response.read())

        print("Done")
        return True

    except Exception as e:
        print(f"Failed: {e}")
        return False


def check_and_download_data(data_dir="data"):
    """
    Check if data files exist, download if missing.

    Args:
        data_dir: Directory containing data files

    Returns:
        bool: True if all files present or downloaded successfully
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # Persnal Google Drive where JHU U.S. COVID-19 and mobility data is stored
    # Note these are the IDs for each dataset
    # From: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA
    drive_files = {
        "cases.csv": "1ZfZtoV3PpZblZYES0A5LHCwp54cR8RJL",
        "deaths.csv": "1kYC9nrCnKbNpnoZKz8o6TDMM371gyxbl",
        "mobility.csv": "1TMqG8Z8vbxmQAv1rNKczYYPCzwT4ZS_q",
    }

    # Check which files exist
    existing_files = []
    missing_files = []

    for filename in drive_files.keys():
        file_path = data_path / filename
        if file_path.exists():
            existing_files.append(filename)
            print(f"Found: {filename}")
        else:
            missing_files.append(filename)

    if not missing_files:
        print("\nAll data files present!")
        return True

    print(f"\nMissing files: {', '.join(missing_files)}")

    # Try to download missing files
    print("\nAttempting to download from Google Drive...")
    downloaded = []
    failed = []

    for filename in missing_files:
        file_id = drive_files[filename]
        file_path = data_path / filename

        if file_id:
            # Try automated download
            if download_file_from_google_drive(file_id, file_path):
                downloaded.append(filename)
            else:
                failed.append(filename)
        else:
            # No file ID, needs manual download
            failed.append(filename)

    # Show results
    if downloaded:
        print(f"\nSuccessfully downloaded: {', '.join(downloaded)}")

    if failed:
        print("\nSome files need manual download:")
        print(
            "Visit: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA"
        )

        file_mapping = {
            "cases.csv": "time_series_covid19_confirmed_US.csv",
            "deaths.csv": "time_series_covid19_deaths_US.csv",
            "mobility.csv": "mobility_report_US.csv",
        }

        print("\nDownload these files and save to 'data/' directory:")
        for local_name in failed:
            if local_name in file_mapping:
                drive_name = file_mapping[local_name]
                print(f"  - {drive_name} → rename to '{local_name}'")

        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("COVID-19 Data Setup")
    print("=" * 60)
    print()

    success = check_and_download_data()

    print()
    print("=" * 60)
    if success:
        print("Ready to run notebooks!")
    else:
        print("Please download the data files as instructed above")
    print("=" * 60)
