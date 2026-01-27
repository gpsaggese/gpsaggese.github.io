"""
Simple data downloader for COVID-19 datasets.

Downloads data from Google Drive if not present locally.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.GluonTS_utils_data_download as ttgcpgudd
"""

import logging
import urllib.request
from pathlib import Path

_LOG = logging.getLogger(__name__)


def _get_confirm_token(response):
    """
    Extract confirmation token from response headers.

    :param response: HTTP response object
    :return: Confirmation token or None
    """
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            if "download_warning" in value:
                return value.split("download_warning=")[1].split(";")[0]
    return None


def download_file_from_google_drive(
    file_id: str,
    destination: Path,
) -> bool:
    """
    Download file from Google Drive.

    :param file_id: Google Drive file ID
    :param destination: Local path to save file
    :return: True if successful, False otherwise
    """
    _LOG.info("Downloading %s... ", destination.name)
    try:
        URL = "https://drive.google.com/uc?export=download"
        session = urllib.request.build_opener()
        # Initial request.
        response = session.open(f"{URL}&id={file_id}")
        token = _get_confirm_token(response)
        # If large file, need confirmation token.
        if token:
            response = session.open(f"{URL}&id={file_id}&confirm={token}")
        # Save file.
        with open(destination, "wb") as f:
            f.write(response.read())
        _LOG.info("Done")
        return True
    except Exception as e:
        _LOG.error("Failed: %s", e)
        return False


def check_and_download_data(
    *,
    data_dir: str = "data",
) -> bool:
    """
    Check if data files exist, download if missing.

    :param data_dir: Directory containing data files
    :return: True if all files present or downloaded successfully
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    # Personal Google Drive where JHU U.S. COVID-19 and mobility data is stored.
    # Note these are the IDs for each dataset.
    # From: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA
    drive_files = {
        "cases.csv": "1ZfZtoV3PpZblZYES0A5LHCwp54cR8RJL",
        "deaths.csv": "1kYC9nrCnKbNpnoZKz8o6TDMM371gyxbl",
        "mobility.csv": "1TMqG8Z8vbxmQAv1rNKczYYPCzwT4ZS_q",
    }
    # Check which files exist.
    existing_files = []
    missing_files = []
    for filename in drive_files.keys():
        file_path = data_path / filename
        if file_path.exists():
            existing_files.append(filename)
            _LOG.info("Found: %s", filename)
        else:
            missing_files.append(filename)
    if not missing_files:
        _LOG.info("\nAll data files present!")
        return True
    _LOG.info("\nMissing files: %s", ", ".join(missing_files))
    # Try to download missing files.
    _LOG.info("\nAttempting to download from Google Drive...")
    downloaded = []
    failed = []
    for filename in missing_files:
        file_id = drive_files[filename]
        file_path = data_path / filename
        if file_id:
            # Try automated download.
            if download_file_from_google_drive(file_id, file_path):
                downloaded.append(filename)
            else:
                failed.append(filename)
        else:
            # No file ID, needs manual download.
            failed.append(filename)
    # Show results.
    if downloaded:
        _LOG.info("\nSuccessfully downloaded: %s", ", ".join(downloaded))
    if failed:
        _LOG.info("\nSome files need manual download:")
        _LOG.info(
            "Visit: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA"
        )
        file_mapping = {
            "cases.csv": "time_series_covid19_confirmed_US.csv",
            "deaths.csv": "time_series_covid19_deaths_US.csv",
            "mobility.csv": "mobility_report_US.csv",
        }
        _LOG.info("\nDownload these files and save to 'data/' directory:")
        for local_name in failed:
            if local_name in file_mapping:
                drive_name = file_mapping[local_name]
                _LOG.info("  - %s → rename to '%s'", drive_name, local_name)
        return False
    return True


def _main() -> None:
    _LOG.info("=" * 60)
    _LOG.info("COVID-19 Data Setup")
    _LOG.info("=" * 60)
    _LOG.info("")
    success = check_and_download_data()
    _LOG.info("")
    _LOG.info("=" * 60)
    if success:
        _LOG.info("Ready to run notebooks!")
    else:
        _LOG.info("Please download the data files as instructed above")
    _LOG.info("=" * 60)


if __name__ == "__main__":
    _main()
