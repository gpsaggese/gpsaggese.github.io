import luigi
import requests
import yfinance as yf
import pandas as pd
import hashlib
import json
import time

from pathlib import Path

from src.config import (
    SEC_DIR,
    PRICE_DIR,
    USER_AGENT,
    COMPANY_FILE,
    START_DATE,
    END_DATE
)
def verify_sha256(file_path, checksum_path):
    """Verify a file against its saved SHA-256 checksum."""
    actual_hash = generate_sha256(file_path)

    with open(checksum_path, "r") as f:
        expected_hash = f.read().strip()

    return actual_hash == expected_hash
# 1. Fetch SEC Filings

class FetchFilings(luigi.Task):

    ticker = luigi.Parameter()
    cik = luigi.Parameter()

    def output(self):

        return luigi.LocalTarget(
            SEC_DIR / f"{self.ticker}.json.sha256"
        )

    def run(self):

        SEC_DIR.mkdir(parents=True, exist_ok=True)

        headers = {
            "User-Agent": USER_AGENT
        }

        # SEC submissions endpoint
        url = (
            "https://data.sec.gov/submissions/"
            f"CIK{self.cik.zfill(10)}.json"
        )

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        # Retrieve historical filing pages
        historical = []

        older_files = (
            data.get("filings", {})
                .get("files", [])
        )

        for file_info in older_files:

            filename = file_info["name"]

            historical_url = (
                "https://data.sec.gov/submissions/"
                + filename
            )

            time.sleep(0.12)

            old_response = requests.get(
                historical_url,
                headers=headers,
                timeout=30
            )

            old_response.raise_for_status()

            historical.append({
                "filename": filename,
                "data": old_response.json()
            })

        # Combine current and historical data
        result = {
            "ticker": self.ticker,
            "cik": self.cik,
            "current": data,
            "historical": historical
        }

        # Save JSON atomically
        output_path = SEC_DIR / f"{self.ticker}.json"

        temp_path = output_path.with_suffix(".tmp")

        with open(temp_path, "w") as f:
            json.dump(result, f)

        temp_path.replace(output_path)

        # Generate SHA-256 checksum
        checksum = hashlib.sha256(
            output_path.read_bytes()
        ).hexdigest()

        with self.output().open("w") as f:
            f.write(checksum)
        if not verify_sha256(output_path, self.output().path):
            raise ValueError(f"Checksum verification failed: {output_path}")
        print(f"SHA-256 verified: {output_path.name}")

# 2. Fetch Daily Stock Prices

class FetchPrices(luigi.Task):

    ticker = luigi.Parameter()

    def output(self):

        return luigi.LocalTarget(
            PRICE_DIR / f"{self.ticker}.csv.sha256"
        )

    def run(self):

        PRICE_DIR.mkdir(parents=True, exist_ok=True)

        # Download historical prices
        df = yf.download(
            self.ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df.empty:

            raise RuntimeError(
                f"No price data returned for {self.ticker}"
            )

        # Save CSV
        output_path = PRICE_DIR / f"{self.ticker}.csv"

        temp_path = output_path.with_suffix(".tmp")

        df.to_csv(temp_path)

        temp_path.replace(output_path)

        # Generate SHA-256 checksum
        checksum = hashlib.sha256(
            output_path.read_bytes()
        ).hexdigest()

        with self.output().open("w") as f:
            f.write(checksum)
        if not verify_sha256(output_path, self.output().path):
            raise ValueError(f"Checksum verification failed: {output_path}")
        print(f"SHA-256 verified: {output_path.name}")

# 3. Master Ingestion Task

class IngestAll(luigi.WrapperTask):

    def requires(self):

        companies = pd.read_csv(
            COMPANY_FILE,
            dtype=str
        )

        tasks = []

        for _, row in companies.iterrows():

            ticker = row["ticker"]
            cik = row["cik"].zfill(10)

            # SEC task
            tasks.append(
                FetchFilings(
                    ticker=ticker,
                    cik=cik
                )
            )

            # Stock price task
            tasks.append(
                FetchPrices(
                    ticker=ticker
                )
            )

        # Market benchmark
        tasks.append(
            FetchPrices(ticker="SPY")
        )

        return tasks
