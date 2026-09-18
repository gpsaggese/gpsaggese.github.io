"""
Utility functions and tool wrappers for Autogen-based financial analysis agents.

Import as:

import tutorials.Autogen.autogen_utils as tauauuti
"""

import glob
import json
import logging
import os
import re
import typing

import autogen_core
import autogen_core.tools
import autogen_ext.code_executors.local
import autogen_ext.models.openai
import bs4
import chromadb
import chromadb.utils.embedding_functions
import markdownify
import matplotlib.pyplot as plt
import nest_asyncio
import sec_edgar_downloader
import yfinance

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


# #############################################################################
# Setup
# #############################################################################

nest_asyncio.apply()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()


# #############################################################################
# Markdown utilities
# #############################################################################


def clean_markdown(text: str) -> str:
    """
    Clean markdown text by removing TERMINATE markers and fixing table spacing.

    :param text: the raw markdown text to clean
    :return: cleaned markdown text with proper spacing around tables
    """
    text = text.replace("TERMINATE", "").strip()
    # Split into lines and process table spacing.
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table = stripped.startswith("|")
        prev_is_table = result[-1].strip().startswith("|") if result else False
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        next_is_table = next_line.startswith("|")
        # Inject blank line before a table starts.
        if is_table and not prev_is_table:
            result.append("")
        result.append(line)
        # Inject blank line after a table ends.
        if is_table and not next_is_table and next_line != "":
            result.append("")
    # Collapse 3+ consecutive blank lines into 2.
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(result))
    return collapsed.strip()


# #############################################################################
# SEC 10-K utilities
# #############################################################################


def embed_10k_to_chroma(ticker: str, file_path: str) -> str:
    """
    Embed 10-K report text chunks into a ChromaDB collection.

    :param ticker: the stock ticker symbol
    :param file_path: path to the cleaned 10-K text file
    :return: the name of the ChromaDB collection
    """
    ticker = ticker.lower()
    collection_name = f"{ticker}_10k"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_ef,
    )
    if collection.count() > 0:
        _LOG.info("Collection '%s' already exists.", collection_name)
        return collection_name
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunk_size = 2000
    overlap = 200
    chunks = [
        text[i : i + chunk_size]
        for i in range(0, len(text), chunk_size - overlap)
    ]
    _LOG.info("Encoding %d chunks for %s...", len(chunks), ticker.upper())
    collection.add(
        documents=chunks,
        ids=[f"{ticker}_{i}" for i in range(len(chunks))],
        metadatas=[{"ticker": ticker.upper(), "source": "10-K"}] * len(chunks),
    )
    return collection_name


def fetch_and_clean_10k(ticker: str) -> str:
    """
    Download and clean a 10-K filing from SEC EDGAR for the given ticker.

    :param ticker: the stock ticker symbol
    :return: path to the saved cleaned text file
    """
    ticker = ticker.upper()
    dl = sec_edgar_downloader.Downloader("MyCompany", "myemail@example.com")
    dl.get("10-K", ticker, limit=1)
    search_path = os.path.join(
        "sec-edgar-filings", ticker, "10-K", "*", "full-submission.txt"
    )
    files = glob.glob(search_path)
    hdbg.dassert(files, "No 10-K files found for ticker '%s'", ticker)
    raw_path = max(files, key=os.path.getmtime)
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = f.read()
    doc_match = re.search(
        r"<DOCUMENT>\s*<TYPE>10-K.*?\s*<TEXT>(.*?)</TEXT>",
        raw_data,
        re.DOTALL | re.IGNORECASE,
    )
    html_content = doc_match.group(1) if doc_match else raw_data
    soup = bs4.BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "link", "meta"]):
        tag.decompose()
    clean_text = markdownify.markdownify(
        str(soup),
        heading_style="ATX",
        tables=True,
        include_empty_tables=False,
    )
    clean_text = re.sub(r"\n\s*\n", "\n\n", clean_text).replace("\xa0", " ")
    os.makedirs("report_texts", exist_ok=True)
    save_path = f"report_texts/{ticker}_10K_clean.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
    return save_path


# #############################################################################
# Market data functions
# #############################################################################


async def get_live_market_data(
    ticker: typing.Annotated[str, "The stock ticker"],
) -> str:
    """
    Fetch the current market price for the given stock ticker.

    :param ticker: the stock ticker symbol
    :return: formatted string with the current price
    """
    stock = yfinance.Ticker(ticker)
    return f"Current Price for {ticker}: ${stock.fast_info['last_price']:.2f}"


async def get_deep_financials(
    ticker: typing.Annotated[str, "The stock ticker"],
) -> str:
    """
    Fetch detailed financial metrics for the given stock ticker.

    :param ticker: the stock ticker symbol
    :return: JSON string with key financial metrics
    """
    t = yfinance.Ticker(ticker)
    inf = t.info
    metrics = {
        "Price/Earnings (Trailing)": inf.get("trailingPE"),
        "Forward P/E": inf.get("forwardPE"),
        "PEG Ratio": inf.get("pegRatio"),
        "Profit Margin": inf.get("profitMargins"),
        "Revenue Growth (YoY)": inf.get("revenueGrowth"),
        "Debt to Equity": inf.get("debtToEquity"),
    }
    clean_metrics = {k: v for k, v in metrics.items() if v is not None}
    return f"Financials for {ticker}: {json.dumps(clean_metrics)}"


async def plot_stock_trend(ticker: str, days: int) -> autogen_core.Image:
    """
    Plot a stock price trend chart and return it as an autogen Image.

    :param ticker: the stock ticker symbol
    :param days: number of days to plot
    :return: autogen Image of the stock chart
    """
    data = yfinance.download(
        ticker, period=f"{days}d", progress=False, multi_level_index=False
    )
    plt.figure(figsize=(10, 5))
    plt.plot(data["Close"], color="#1f77b4")
    plt.title(f"{ticker} - {days} Days")
    filename = f"{ticker}_chart.png"
    plt.savefig(filename)
    plt.close()
    return autogen_core.Image.from_file(filename)


async def query_10k_report(
    ticker: typing.Annotated[str, "Ticker"],
    question: typing.Annotated[str, "Question"],
) -> str:
    """
    Query the ChromaDB collection for relevant sections from a 10-K report.

    :param ticker: the stock ticker symbol
    :param question: the question to answer from the 10-K report
    :return: formatted string with relevant evidence from the 10-K
    """
    col_name = f"{ticker.lower()}_10k"
    collection = chroma_client.get_collection(
        name=col_name, embedding_function=default_ef
    )
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n---\n\n".join(results["documents"][0])
    return f"### Evidence from {ticker} 10-K:\n{context}"


# #############################################################################
# Tool wrappers
# #############################################################################

os.makedirs("coding", exist_ok=True)
code_executor = autogen_ext.code_executors.local.LocalCommandLineCodeExecutor(
    work_dir="coding"
)

market_tool = autogen_core.tools.FunctionTool(
    get_live_market_data, description="Get price"
)
plot_tool = autogen_core.tools.FunctionTool(
    plot_stock_trend, description="Generates chart"
)
financial_tool = autogen_core.tools.FunctionTool(
    get_deep_financials, description="Gets financials"
)
rag_search_tool = autogen_core.tools.FunctionTool(
    query_10k_report, description="Summarize the annual 10-K report."
)
