import os
import zipfile
import urllib.request
from io import BytesIO

URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
RAW_FILENAME = "training.1600000.processed.noemoticon.csv"
OUTPUT_FILENAME = "Sentiment140_raw.csv"

def download_and_extract():
    print("Downloading Sentiment140 dataset...")

    with urllib.request.urlopen(URL) as response:
        data = response.read()

    with zipfile.ZipFile(BytesIO(data)) as z:
        if RAW_FILENAME not in z.namelist():
            raise FileNotFoundError(f"{RAW_FILENAME} not found in archive")

        print(f"Extracting and saving as {OUTPUT_FILENAME}...")
        with z.open(RAW_FILENAME) as src, open(OUTPUT_FILENAME, "wb") as dst:
            dst.write(src.read())

    print("Done.")
    print(f"Saved at: {os.path.abspath(OUTPUT_FILENAME)}")

if __name__ == "__main__":
    download_and_extract()