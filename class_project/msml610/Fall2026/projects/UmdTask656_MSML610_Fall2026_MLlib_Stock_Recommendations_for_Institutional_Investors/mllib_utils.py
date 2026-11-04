"""
Utilities for the MLlib 13F stock-recommendation project.

Import as:

import mllib_utils
"""

import logging
import os
import shutil
import urllib.request
import zipfile
from typing import List, Optional

import pyspark.sql
import pyspark.sql.functions as F
import pyspark.sql.window as W

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo


_LOG = logging.getLogger(__name__)

# SEC blocks clients that don't send a User-Agent.
_SEC_USER_AGENT = "UMD MSML610 educational project msml610@umd.edu"
_SEC_FILES = "https://www.sec.gov/files/"

# Zip path under sec.gov/files, and the quarter-end that window reports.
# Checked 2026-11-04. Sep-Nov 2026 was not posted yet. The newest zip
# sits in a different directory than the other three.
_QUARTERS = [
    (
        "structureddata/data/form-13f-data-sets/"
        "01sep2025-30nov2025_form13f.zip",
        "2025-09-30",
    ),
    (
        "structureddata/data/form-13f-data-sets/"
        "01dec2025-28feb2026_form13f.zip",
        "2025-12-31",
    ),
    (
        "structureddata/data/form-13f-data-sets/"
        "01mar2026-31may2026_form13f.zip",
        "2026-03-31",
    ),
    (
        "datastandardsinnovation/data/form-13f-data-sets/"
        "01jun2026-31aug2026_form13f.zip",
        "2026-06-30",
    ),
]

# Project cap so the manager-by-stock matrix fits on one machine.
_N_MANAGERS = 2000
_N_CUSIPS = 5000

# Only the columns the join needs. Header names, not file position.
_COLUMNS = {
    "SUBMISSION": [
        "ACCESSION_NUMBER",
        "FILING_DATE",
        "SUBMISSIONTYPE",
        "CIK",
        "PERIODOFREPORT",
    ],
    "COVERPAGE": [
        "ACCESSION_NUMBER",
        "AMENDMENTNO",
        "AMENDMENTTYPE",
        "FILINGMANAGER_NAME",
    ],
    "INFOTABLE": [
        "ACCESSION_NUMBER",
        "NAMEOFISSUER",
        "CUSIP",
        "VALUE",
        "SSHPRNAMTTYPE",
        "PUTCALL",
    ],
}

# File dates look like 31-DEC-2025. The container locale has no month names.
_MONTHS = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into this module.
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


def project_data_dir() -> str:
    """
    Return the `data` directory next to this file.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_spark() -> pyspark.sql.SparkSession:
    """
    Local Spark with a 4 GB driver. Docker Desktop needs about 8 GB.
    """
    active = pyspark.sql.SparkSession.getActiveSession()
    if active is not None:
        return active
    # 4g heap. Give Docker Desktop about 8 GB so this fits.
    session = (
        pyspark.sql.SparkSession.builder.appName("mllib_13f_holdings")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    return session


def _sec_date(column: str) -> pyspark.sql.Column:
    """
    Turn a `DD-MON-YYYY` string into a date.
    """
    parts = F.split(F.upper(F.trim(F.col(column))), "-")
    month = F.lit(None).cast("string")
    for name, number in _MONTHS.items():
        month = F.when(parts.getItem(1) == name, number).otherwise(month)
    return F.to_date(
        F.concat_ws(
            "-",
            parts.getItem(2),
            month,
            F.lpad(parts.getItem(0), 2, "0"),
        ),
        "yyyy-MM-dd",
    )


def _quarter_dirs(data_dir: str) -> List[str]:
    """
    Download the four zips if needed and unzip the three TSVs.
    """
    zip_dir = os.path.join(data_dir, "zips")
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(zip_dir, exist_ok=True)
    quarter_dirs = []
    for path, _period in _QUARTERS:
        file_name = os.path.basename(path)
        zip_path = os.path.join(zip_dir, file_name)
        # Reuse a zip that is already on disk.
        have_zip = os.path.isfile(zip_path) and os.path.getsize(zip_path) > 0
        if have_zip:
            with open(zip_path, "rb") as handle:
                have_zip = handle.read(2) == b"PK"
        if not have_zip:
            url = _SEC_FILES + path
            _LOG.info("Downloading '%s'", url)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": _SEC_USER_AGENT},
            )
            partial_path = zip_path + ".partial"
            with urllib.request.urlopen(request) as response:
                with open(partial_path, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            os.replace(partial_path, zip_path)
        quarter_id = file_name.replace("_form13f.zip", "")
        quarter_dir = os.path.join(raw_dir, quarter_id)
        os.makedirs(quarter_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            for table in _COLUMNS:
                dest_path = os.path.join(quarter_dir, table + ".tsv")
                if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
                    continue
                _LOG.info("Extracting '%s' from '%s'", table, file_name)
                with archive.open(table + ".tsv") as src:
                    with open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        quarter_dirs.append(quarter_dir)
    return quarter_dirs


def _load(spark, quarter_dirs, table):
    """
    Read one TSV from each quarter and stack them.
    """
    frames = []
    for quarter_dir in quarter_dirs:
        # Cover-page notes are quoted. They have no tabs, so ignore quotes.
        frame = (
            spark.read.option("header", True)
            .option("sep", "\t")
            .option("quote", "\u0000")
            .option("escape", "\u0000")
            .option("nullValue", "")
            .csv(os.path.join(quarter_dir, table + ".tsv"))
            .select(*_COLUMNS[table])
        )
        if table == "INFOTABLE":
            # Since 2023 this column is dollars, not thousands.
            frame = frame.withColumn("VALUE", F.col("VALUE").cast("long"))
        frames.append(frame)
    stacked = frames[0]
    for frame in frames[1:]:
        stacked = stacked.unionByName(frame)
    return stacked


def _holdings(spark, quarter_dirs):
    """
    Join manager, quarter, and CUSIP, then keep the top managers and stocks.
    """
    submission = _load(spark, quarter_dirs, "SUBMISSION")
    coverpage = _load(spark, quarter_dirs, "COVERPAGE")
    # Accession number is the filing id. A duplicate would double the join.
    submission = submission.dropDuplicates(["ACCESSION_NUMBER"])
    coverpage = coverpage.dropDuplicates(["ACCESSION_NUMBER"])
    filings = submission.join(coverpage, "ACCESSION_NUMBER")
    # 13F-NT is just a notice. It has no stock list.
    filings = filings.filter(F.col("SUBMISSIONTYPE").isin("13F-HR", "13F-HR/A"))
    filings = filings.withColumn("filing_date", _sec_date("FILING_DATE"))
    filings = filings.withColumn("report_period", _sec_date("PERIODOFREPORT"))
    filings = filings.withColumn("cik", F.trim(F.col("CIK")))
    filings = filings.withColumn(
        "manager_name",
        F.trim(F.col("FILINGMANAGER_NAME")),
    )
    # Each zip also has late fixes of older quarters. Those are incomplete.
    periods = [period for _path, period in _QUARTERS]
    filings = filings.filter(F.col("report_period").cast("string").isin(periods))
    # A restatement replaces the whole report. "NEW HOLDINGS" only adds
    # rows, so a later one of those is kept along with the full filing.
    amendment = F.upper(F.trim(F.coalesce(F.col("AMENDMENTTYPE"), F.lit(""))))
    filings = filings.withColumn(
        "amendment_no",
        F.coalesce(F.col("AMENDMENTNO").cast("int"), F.lit(0)),
    )
    filings = filings.withColumn("is_new", amendment == "NEW HOLDINGS")
    order = W.Window.partitionBy("cik", "report_period").orderBy(
        F.desc("filing_date"),
        F.desc("amendment_no"),
        F.desc("ACCESSION_NUMBER"),
    )
    latest = (
        filings.filter(~F.col("is_new"))
        .withColumn("rn", F.row_number().over(order))
        .filter(F.col("rn") == 1)
    )
    adds = filings.filter(F.col("is_new")).drop("manager_name")
    adds = adds.join(
        latest.select(
            "cik",
            "report_period",
            "manager_name",
            F.col("filing_date").alias("base_date"),
            F.col("amendment_no").alias("base_no"),
        ),
        ["cik", "report_period"],
    )
    adds = adds.filter(
        (F.col("filing_date") > F.col("base_date"))
        | (
            (F.col("filing_date") == F.col("base_date"))
            & (F.col("amendment_no") > F.col("base_no"))
        )
    )
    keys = latest.select(
        "ACCESSION_NUMBER",
        "cik",
        "manager_name",
        "report_period",
    ).unionByName(
        adds.select("ACCESSION_NUMBER", "cik", "manager_name", "report_period")
    )
    # Puts and calls use the stock's CUSIP, so drop them. SH = shares.
    info = _load(spark, quarter_dirs, "INFOTABLE")
    kept = keys.select("ACCESSION_NUMBER").distinct()
    info = info.join(kept, "ACCESSION_NUMBER")
    info = info.withColumn("cusip", F.upper(F.trim(F.col("CUSIP"))))
    putcall = F.upper(F.trim(F.coalesce(F.col("PUTCALL"), F.lit(""))))
    share_type = F.upper(F.trim(F.coalesce(F.col("SSHPRNAMTTYPE"), F.lit(""))))
    info = info.filter(
        (putcall == "")
        & (share_type == "SH")
        & F.col("cusip").rlike("^[0-9A-Z]{9}$")
        & F.col("VALUE").isNotNull()
        & (F.col("VALUE") > 0)
    )
    # One stock can be listed twice when discretion is split. Add the dollars.
    per_filing = info.groupBy("ACCESSION_NUMBER", "cusip").agg(
        F.sum("VALUE").alias("value"),
        F.max(F.trim(F.col("NAMEOFISSUER"))).alias("issuer_name"),
    )
    holdings = keys.join(per_filing, "ACCESSION_NUMBER")
    holdings = holdings.groupBy(
        "cik",
        "manager_name",
        "report_period",
        "cusip",
    ).agg(
        F.sum("value").cast("double").alias("value"),
        F.max("issuer_name").alias("issuer_name"),
    )
    # Biggest managers first, then the stocks those managers hold most often.
    top_managers = (
        holdings.groupBy("cik")
        .count()
        .orderBy(F.desc("count"), F.asc("cik"))
        .limit(_N_MANAGERS)
    )
    holdings = holdings.join(top_managers.select("cik"), "cik")
    top_cusips = (
        holdings.groupBy("cusip")
        .agg(F.countDistinct("cik").alias("n_managers"))
        .orderBy(F.desc("n_managers"), F.asc("cusip"))
        .limit(_N_CUSIPS)
    )
    holdings = holdings.join(top_cusips.select("cusip"), "cusip")
    return holdings.select(
        "cik",
        "manager_name",
        "report_period",
        "cusip",
        "issuer_name",
        "value",
    )


def _log_counts(holdings) -> None:
    """
    Print row, manager, CUSIP, and quarter counts.
    """
    stats = holdings.agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("cik").alias("n_managers"),
        F.countDistinct("cusip").alias("n_cusips"),
        F.countDistinct("report_period").alias("n_periods"),
    ).collect()[0]
    _LOG.info(
        "Holdings rows=%s managers=%s cusips=%s periods=%s",
        stats["n_rows"],
        stats["n_managers"],
        stats["n_cusips"],
        stats["n_periods"],
    )
    hdbg.dassert_lt(0, stats["n_rows"], "Holdings table is empty")


def ingest_holdings(
    data_dir: str,
    *,
    spark: Optional[pyspark.sql.SparkSession] = None,
    overwrite: bool = False,
) -> pyspark.sql.DataFrame:
    """
    Build the filtered holdings table and write it as Parquet.

    Reads the existing file when it is already there.

    :param data_dir: folder for the zips and the Parquet
    :param spark: session to use. Default is `build_spark`
    :param overwrite: rebuild even if Parquet exists
    :return: `cik`, `manager_name`, `report_period`, `cusip`,
        `issuer_name`, `value` (dollars)
    """
    if spark is None:
        spark = build_spark()
    parquet_dir = os.path.join(data_dir, "holdings")
    ready = os.path.isfile(os.path.join(parquet_dir, "_SUCCESS"))
    if ready and not overwrite:
        _LOG.info("Reading holdings Parquet '%s'", parquet_dir)
        holdings = spark.read.parquet(parquet_dir)
    else:
        os.makedirs(data_dir, exist_ok=True)
        holdings = _holdings(spark, _quarter_dirs(data_dir))
        # Save the join so the count and the write don't both rerun it.
        holdings = holdings.persist()
        _log_counts(holdings)
        _LOG.info("Writing holdings Parquet '%s'", parquet_dir)
        holdings.write.mode("overwrite").parquet(parquet_dir)
        return holdings
    _log_counts(holdings)
    return holdings


if __name__ == "__main__":
    hdbg.init_logger(verbosity=logging.INFO)
    ingest_holdings(project_data_dir())
