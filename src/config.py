from pathlib import Path

# Project directory
BASE_DIR = Path("/app")

# Data directories
DATA_DIR = BASE_DIR / "data"

SEC_DIR = DATA_DIR / "raw" / "sec"
PRICE_DIR = DATA_DIR / "raw" / "prices"

PROCESSED_DIR = DATA_DIR / "processed"

# SEC API identification
USER_AGENT = (
    "EarningsPipeline/1.0 "
    "Your Name your_email@example.com"
)

# Historical price range
START_DATE = "2020-01-01"
END_DATE = "2026-11-01"

# Company mapping
COMPANY_FILE = DATA_DIR / "companies.csv"
