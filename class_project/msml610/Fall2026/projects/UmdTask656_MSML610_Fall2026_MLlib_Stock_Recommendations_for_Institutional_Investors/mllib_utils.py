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

import numpy as np
import pandas as pd
import pyspark.ml.clustering as mlcluster
import pyspark.ml.evaluation as mleval
import pyspark.ml.feature as mlfeat
import pyspark.ml.functions as mlfunc
import pyspark.ml.recommendation as mlrec
import pyspark.ml.tuning as mltune
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


def _windows_hadoop() -> None:
    """
    Give Windows the Hadoop dll PySpark uses to list local files.

    Linux, including the project container, does not need this.
    """
    if os.name != "nt":
        return
    # data/ is gitignored. The two files are the Hadoop 3.3 Windows bin.
    bin_dir = os.path.join(project_data_dir(), "hadoop", "bin")
    os.makedirs(bin_dir, exist_ok=True)
    base = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/"
    for name in ("hadoop.dll", "winutils.exe"):
        dest = os.path.join(bin_dir, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            with open(dest, "rb") as handle:
                if handle.read(2) == b"MZ":
                    continue
        url = base + name
        _LOG.info("Downloading '%s'", url)
        request = urllib.request.Request(
            url, headers={"User-Agent": _SEC_USER_AGENT}
        )
        partial = dest + ".partial"
        with urllib.request.urlopen(request) as response:
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
        os.replace(partial, dest)
    home = os.path.dirname(bin_dir)
    # Has to be set before the Spark JVM starts, or the link error sticks.
    os.environ["HADOOP_HOME"] = home
    os.environ["hadoop.home.dir"] = home
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def build_spark() -> pyspark.sql.SparkSession:
    """
    Local Spark with a 4 GB driver. Docker Desktop needs about 8 GB.
    """
    _windows_hadoop()
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


# #############################################################################
# Example notebook
# #############################################################################

# Tickers are labels. The filing id is the CUSIP.
KNOWN_STOCKS = [
    ("037833100", "AAPL"),
    ("594918104", "MSFT"),
    ("67066G104", "NVDA"),
    ("023135106", "AMZN"),
    ("46625H100", "JPM"),
    ("30231G102", "XOM"),
]


def _periods(ratings):
    rows = ratings.select(F.col("report_period").cast("string").alias("p"))
    periods = [row["p"] for row in rows.distinct().orderBy("p").collect()]
    return list(zip(periods[:-1], periods[1:]))


def _quarter(ratings, period):
    return ratings.filter(F.col("report_period").cast("string") == period)


def _opens(train, test):
    # next quarter, not this quarter, and the manager filed this quarter
    keys = train.select("cik", "cusip").distinct()
    managers = train.select("cik").distinct()
    opened = test.select("cik", "cusip").join(
        keys, ["cik", "cusip"], "left_anti"
    )
    return opened.join(managers, "cik")


def open_summary(ratings) -> pd.DataFrame:
    """
    New positions next quarter.

    cold_share is the part with no item factor: that CUSIP is missing
    from the training quarter, so ALS cannot recommend it.
    """
    rows = []
    for train_period, test_period in _periods(ratings):
        train = _quarter(ratings, train_period)
        opened = _opens(train, _quarter(ratings, test_period))
        seen = train.select("cusip").distinct().withColumn("seen", F.lit(1))
        marked = opened.join(seen, "cusip", "left")
        n_opens = int(marked.count())
        n_cold = int(marked.filter(F.col("seen").isNull()).count())
        median = train.groupBy("cik").count()
        median = median.approxQuantile("count", [0.5], 0.01)[0]
        share = 0.0 if n_opens == 0 else n_cold / float(n_opens)
        rows.append(
            {
                "train_period": train_period,
                "test_period": test_period,
                "median_positions": int(round(median)),
                "n_opens": n_opens,
                "n_cold_opens": n_cold,
                "cold_share": round(share, 4),
            }
        )
    return pd.DataFrame(rows)


def _ids(train):
    # most common manager and CUSIP get id 0
    for src, dst in (("cik", "manager_idx"), ("cusip", "item_idx")):
        indexer = mlfeat.StringIndexer(inputCol=src, outputCol=dst)
        train = indexer.fit(train).transform(train)
    return train.select(
        F.col("manager_idx").cast("int").alias("manager_idx"),
        F.col("item_idx").cast("int").alias("item_idx"),
        "rating",
        "cik",
        "cusip",
        "issuer_name",
        "value",
    )


def _matrix(factors):
    frame = factors.select(F.col("id").cast("int").alias("id"), "features")
    frame = frame.toPandas().sort_values("id")
    matrix = np.vstack([np.asarray(v, dtype=float) for v in frame["features"]])
    return matrix


def _top10(row, held):
    # drop what they already hold. a typical book is ~500 names
    scores = row.copy()
    if held:
        scores[np.asarray(held, dtype=int)] = -np.inf
    n_take = min(10, int(np.isfinite(scores).sum()))
    if n_take == 0:
        return []
    pick = np.argpartition(scores, -n_take)[-n_take:]
    pick = pick[np.argsort(-scores[pick])]
    return [int(item) for item in pick]


def _lists(spark, recs):
    rows = [
        (int(manager), [float(item) for item in items])
        for manager, items in recs.items()
        if items
    ]
    frame = spark.createDataFrame(rows, ["manager_idx", "pred_items"])
    return frame.select(
        "manager_idx",
        F.expr("transform(pred_items, x -> cast(x as double))").alias(
            "pred_items"
        ),
    )


def _metrics(frame):
    scores = {}
    names = (("precisionAtK", "precision_at_10"), ("ndcgAtK", "ndcg_at_10"))
    for spark_name, column in names:
        evaluator = mleval.RankingEvaluator(
            predictionCol="pred_items",
            labelCol="true_items",
            metricName=spark_name,
            k=10,
        )
        scores[column] = round(float(evaluator.evaluate(frame)), 4)
    return scores


def score_quarter(
    ratings, train_period, test_period, rank, reg_param, alpha, max_iter=10
):
    """
    Fit implicit ALS on one quarter and score the next quarter's opens.
    """
    train = _quarter(ratings, train_period).persist()
    train.count()
    opened = _opens(train, _quarter(ratings, test_period))
    seen = train.select("cusip").distinct().withColumn("seen", F.lit(1))
    marked = opened.join(seen, "cusip", "left")
    n_opens = int(marked.count())
    n_cold = int(marked.filter(F.col("seen").isNull()).count())
    cold_share = 0.0 if n_opens == 0 else round(n_cold / n_opens, 4)
    # no factor if nobody held the CUSIP this quarter
    keep = marked.filter(F.col("seen").isNotNull()).select("cik", "cusip")
    indexed = _ids(train).persist()
    indexed.count()
    n_users = int(indexed.agg(F.max("manager_idx")).first()[0]) + 1
    n_items = int(indexed.agg(F.max("item_idx")).first()[0]) + 1
    # common spelling, so a typo like "PPLE" doesn't win
    counts = indexed.groupBy("item_idx", "cusip", "issuer_name").count()
    order = W.Window.partitionBy("item_idx").orderBy(
        F.desc("count"), F.asc("issuer_name")
    )
    names = counts.withColumn("rn", F.row_number().over(order))
    names = names.filter(F.col("rn") == 1).drop("rn", "count")
    holders = indexed.groupBy("item_idx").agg(
        F.countDistinct("manager_idx").alias("n_managers")
    )
    items = names.join(holders, "item_idx").toPandas()
    held_rows = indexed.select("manager_idx", "item_idx").toPandas()
    held = [[] for _ in range(n_users)]
    for manager_idx, item_idx in zip(
        held_rows["manager_idx"], held_rows["item_idx"]
    ):
        held[int(manager_idx)].append(int(item_idx))
    labels = keep.join(indexed.select("cik", "manager_idx").distinct(), "cik")
    labels = labels.join(indexed.select("cusip", "item_idx").distinct(), "cusip")
    labels = labels.groupBy("manager_idx").agg(
        F.collect_set(F.col("item_idx").cast("double")).alias("true_items")
    )
    # terciles of book value
    aum = indexed.groupBy("manager_idx").agg(F.sum("value").alias("aum"))
    aum = aum.toPandas().sort_values(["aum", "manager_idx"])
    aum["bucket"] = pd.qcut(
        aum["aum"].rank(method="first"), 3, labels=["small", "mid", "large"]
    )
    pairs = zip(aum["manager_idx"].astype(int), aum["bucket"].astype(str))
    buckets = list(pairs)
    buckets = indexed.sparkSession.createDataFrame(
        buckets, ["manager_idx", "bucket"]
    )
    labels = labels.join(buckets, "manager_idx")
    managers = [int(row["manager_idx"]) for row in labels.collect()]
    model = mlrec.ALS(
        userCol="manager_idx",
        itemCol="item_idx",
        ratingCol="rating",
        implicitPrefs=True,
        coldStartStrategy="drop",
        rank=int(rank),
        maxIter=int(max_iter),
        regParam=float(reg_param),
        alpha=float(alpha),
        numUserBlocks=min(10, n_users),
        numItemBlocks=min(10, n_items),
        seed=7,
    ).fit(indexed.select("manager_idx", "item_idx", "rating"))
    # same score recommendForAllUsers uses
    dots = _matrix(model.userFactors) @ _matrix(model.itemFactors).T
    als_recs = {
        manager: _top10(dots[manager], held[manager]) for manager in managers
    }
    weight = np.zeros(n_items)
    for row in items.itertuples(index=False):
        weight[int(row.item_idx)] = row.n_managers
    popular = list(np.argsort(-(weight - np.arange(n_items) * 1e-6)))
    pop_recs = {}
    for manager in managers:
        banned = set(held[manager])
        pop_recs[manager] = [i for i in popular if i not in banned][:10]
    rows = []
    spark = indexed.sparkSession
    for model_name, recs in (("ALS", als_recs), ("most held", pop_recs)):
        joined = labels.join(_lists(spark, recs), "manager_idx")
        for bucket in ("all", "small", "mid", "large"):
            if bucket == "all":
                subset = joined
            else:
                subset = joined.filter(F.col("bucket") == bucket)
            n_managers = subset.count()
            scores = _metrics(subset) if n_managers else {}
            rows.append(
                {
                    "train_period": train_period,
                    "test_period": test_period,
                    "model": model_name,
                    "bucket": bucket,
                    "precision_at_10": scores.get(
                        "precision_at_10", float("nan")
                    ),
                    "ndcg_at_10": scores.get("ndcg_at_10", float("nan")),
                    "n_managers": int(n_managers),
                    "cold_share": cold_share,
                    "rank": int(rank),
                    "reg_param": float(reg_param),
                    "alpha": float(alpha),
                }
            )
    indexed.unpersist()
    train.unpersist()
    return pd.DataFrame(rows), model, items


def tune_als(ratings, ranks, reg_params, alphas, max_iter=10):
    """
    Grid on the earliest pair. First row is the best NDCG@10.
    """
    train_period, test_period = _periods(ratings)[0]
    # ParamGridBuilder only builds the list. Each one is fit below.
    als = mlrec.ALS(rank=int(ranks[0]), regParam=float(reg_params[0]), alpha=1.0)
    grid = mltune.ParamGridBuilder()
    grid = grid.addGrid(als.rank, [int(v) for v in ranks])
    grid = grid.addGrid(als.regParam, [float(v) for v in reg_params])
    grid = grid.addGrid(als.alpha, [float(v) for v in alphas])
    rows = []
    for params in grid.build():
        metrics, _model, _items = score_quarter(
            ratings,
            train_period,
            test_period,
            int(params[als.rank]),
            float(params[als.regParam]),
            float(params[als.alpha]),
            max_iter=max_iter,
        )
        hit = metrics[(metrics.model == "ALS") & (metrics.bucket == "all")]
        rows.append(hit.iloc[0].to_dict())
    table = pd.DataFrame(rows).sort_values(
        ["ndcg_at_10", "precision_at_10", "rank"],
        ascending=[False, False, True],
    )
    return table.reset_index(drop=True)


def score_quarters(ratings, rank, reg_param, alpha, max_iter=10):
    """
    Score one setting on every pair. The model is the latest quarter.
    """
    frames = []
    model, items = None, None
    for index, (train_period, test_period) in enumerate(_periods(ratings)):
        metrics, model, items = score_quarter(
            ratings,
            train_period,
            test_period,
            rank,
            reg_param,
            alpha,
            max_iter=max_iter,
        )
        metrics = metrics.copy()
        metrics["role"] = "tuned" if index == 0 else "held_out"
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True), model, items


def item_neighbors(model, items, seeds, n=5):
    """
    Closest names by cosine similarity of the item factors.
    """
    matrix = _matrix(model.itemFactors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(norms, 1e-12)
    by_cusip = items.drop_duplicates("cusip").set_index("cusip")
    by_id = items.drop_duplicates("item_idx").set_index("item_idx")
    rows = []
    for cusip, ticker in seeds:
        if cusip not in by_cusip.index:
            continue
        item_idx = int(by_cusip.loc[cusip, "item_idx"])
        scores = unit @ unit[item_idx]
        scores[item_idx] = -np.inf
        for rank, neighbor in enumerate(np.argsort(-scores)[:n], start=1):
            rows.append(
                {
                    "seed": ticker,
                    "rank": rank,
                    "neighbor": str(by_id.loc[int(neighbor), "issuer_name"]),
                    "cosine": round(float(scores[neighbor]), 3),
                }
            )
    return pd.DataFrame(rows)


def cluster_items(model, items, k=8):
    """
    KMeans on the item factors, plus x/y for a scatter plot.
    """
    # stored as a float array. KMeans wants a vector
    factors = model.itemFactors.withColumn(
        "features", mlfunc.array_to_vector(F.col("features"))
    )
    kmeans = mlcluster.KMeans(k=int(k), seed=7, predictionCol="cluster")
    clustered = kmeans.fit(factors).transform(factors)
    pca = mlfeat.PCA(k=2, inputCol="features", outputCol="xy")
    projected = pca.fit(clustered).transform(clustered)
    points = projected.select(
        F.col("id").cast("int").alias("item_idx"), "cluster", "xy"
    ).toPandas()
    points["x"] = points["xy"].map(lambda value: float(value[0]))
    points["y"] = points["xy"].map(lambda value: float(value[1]))
    points = points.drop(columns="xy").merge(items, on="item_idx")
    ranked = points.sort_values(
        ["cluster", "n_managers"], ascending=[True, False]
    )
    top = ranked.drop_duplicates(["cluster", "issuer_name"])
    top = top.groupby("cluster").head(5)
    examples = top.groupby("cluster")["issuer_name"].agg(
        lambda names: "; ".join(names)
    )
    counts = points.groupby("cluster").size().rename("n")
    summary = counts.to_frame().join(examples.rename("examples"))
    return summary.reset_index(), points


if __name__ == "__main__":
    hdbg.init_logger(verbosity=logging.INFO)
    ingest_holdings(project_data_dir())
