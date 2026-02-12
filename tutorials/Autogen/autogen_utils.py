import os 
import asyncio
import logging
import json
import yfinance as yf
import matplotlib.pyplot as plt
import nest_asyncio
from typing import Annotated
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_core import Image
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent
from sec_edgar_downloader import Downloader
import PyPDF2
# --- 1. THE "SILENCE" PROTOCOL ---
nest_asyncio.apply()
logging.disable(logging.CRITICAL) 

# --- 2. THE TOOLS ---

async def get_live_market_data(ticker: Annotated[str, "The stock ticker"]) -> str:
    try:
        stock = yf.Ticker(ticker)
        return f"Current Price for {ticker}: ${stock.fast_info['last_price']:.2f}"
    except:
        return "Price fetch failed."

async def get_deep_financials(ticker: Annotated[str, "The stock ticker"]) -> str:
    try:
        t = yf.Ticker(ticker)
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
    except Exception:
        return "Financial data fetch failed."

async def plot_stock_trend(ticker: Annotated[str, "Ticker symbol"], days: Annotated[int, "Number of days"]) -> Image:
    try:
        data = yf.download(ticker, period=f"{days}d", progress=False, multi_level_index=False)
        plt.figure(figsize=(10, 5))
        plt.plot(data['Close'], color='#1f77b4')
        plt.title(f"{ticker} - {days} Days")
        filename = f"{ticker}_chart.png"
        plt.savefig(filename)
        plt.close()
        return Image.from_file(filename)
    except Exception:
        return "Plotting failed."

async def read_10k_report(file_path: str) -> str:
    """Reads the text from a local PDF 10-K report."""
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages[:10]: # Read first 10 pages for demo
            text += page.extract_text()
    return text[:2000] # Return a chunk for the LLM

dl = Downloader("MyCompany", "myemail@example.com")

async def fetch_latest_10k(ticker: Annotated[str, "The stock ticker"]) -> str:
    """Fetches the latest 10-K (Annual Report) from the SEC for a given ticker."""
    try:
        # Downloads to a folder named 'sec-edgar-filings'
        dl.get("10-K", ticker, limit=1, download_details=True)
        
        # Path logic to find the downloaded file (SEC creates nested folders)
        base_path = f"sec-edgar-filings/{ticker}/10-K"
        latest_folder = sorted(os.listdir(base_path))[-1]
        file_path = os.path.join(base_path, latest_folder, "full-submission.txt")
        
        return f"Successfully downloaded latest 10-K for {ticker} to {file_path}"
    except Exception as e:
        return f"Failed to fetch 10-K: {str(e)}"

async def convert_sec_to_text(ticker: Annotated[str, "The stock ticker"]) -> str:
    """Cleans a downloaded SEC submission file into a standard text file for analysis."""
    try:
        base_path = f"sec-edgar-filings/{ticker}/10-K"
        if not os.path.exists(base_path):
            return "No downloaded filings found. Run fetch_latest_10k first."

        # Find the latest downloaded file
        latest_folder = sorted(os.listdir(base_path))[-1]
        input_path = os.path.join(base_path, latest_folder, "full-submission.txt")
        
        output_dir = "report_texts"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ticker}_10K_clean.txt")

        # Basic cleaning logic
        with open(input_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        # We take a large chunk, but skip common SEC header metadata
        clean_text = raw_content[raw_content.find("<DOCUMENT>"):] 
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text[:50000]) # Save a healthy chunk for the agents

        return f"Converted and cleaned {ticker} report. Saved to {output_path}"
    except Exception as e:
        return f"Conversion failed: {str(e)}"

# Define the tool for the Librarian
pdf_converter_tool = FunctionTool(
    convert_sec_to_text, 
    description="Converts raw SEC filings into readable text files."
)

# --- NEW COMPONENT: Code Execution ---
# This creates a folder to run the Python code safely
os.makedirs("coding", exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

pdf_tool = FunctionTool(read_10k_report, description="Reads annual report PDF")

market_tool = FunctionTool(get_live_market_data, description="Get price")
plot_tool = FunctionTool(plot_stock_trend, description="Generates chart")
financial_tool = FunctionTool(get_deep_financials, description="Gets financials")
fetch_tool = FunctionTool(fetch_latest_10k, description="Downloads the latest annual report from the SEC.")


