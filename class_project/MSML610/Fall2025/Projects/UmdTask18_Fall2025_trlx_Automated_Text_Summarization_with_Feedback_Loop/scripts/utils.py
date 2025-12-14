"""
RLHF News Summarization System - Utility Functions

This module provides a clean API for the modular news summarization system
using the DPO-trained T5-large model.

Example Usage:
    from scripts.utils import summarize_text, summarize_url, summarize_file, refine_summary
    
    # Summarize text
    result = summarize_text("Your article text here...")
    print(result["summary"])
    
    # Summarize from URL
    result = summarize_url("https://example.com/article")
    print(result["summary"])
    
    # Summarize PDF/DOCX
    result = summarize_file("document.pdf")
    print(result["summary"])
    
    # Refine summary
    refined = refine_summary(result["summary"], "make it shorter")
    print(refined["refined_summary"])
"""

import sys
from pathlib import Path

# Add scripts directory to path (we're already in scripts/)
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from typing import Optional, List, Dict, Union
import logging

# Import pipeline
from pipeline.summarization_pipeline import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def summarize_text(
    text: str,
    instructions: Optional[str] = None,
    clean_text: bool = True
) -> Dict:
    """
    Summarize raw text using DPO T5-large model.
    
    This function handles long texts by:
    1. Cleaning the text (optional)
    2. Splitting into sentence-aware chunks (900 tokens each)
    3. Summarizing each chunk independently
    4. Hierarchically aggregating chunk summaries
    
    Args:
        text: Input text to summarize
        instructions: Optional user instructions (e.g., "make it shorter", "bullet points")
        clean_text: Whether to clean and normalize text first
    
    Returns:
        Dictionary containing:
            - summary: Final aggregated summary
            - num_chunks: Number of chunks created
            - chunk_summaries: List of individual chunk summaries
            - input_length: Length of input text
            - summary_length: Length of final summary
    
    Example:
        >>> result = summarize_text("Long article text...", instructions="make it brief")
        >>> print(result["summary"])
    """
    pipeline = get_pipeline()
    return pipeline.summarize_text(text, instructions, clean_text)


def summarize_url(
    url: str,
    instructions: Optional[str] = None
) -> Dict:
    """
    Summarize article from URL.
    
    This function:
    1. Fetches HTML from URL
    2. Extracts clean article text (removes ads, menus, etc.)
    3. Summarizes using the same process as summarize_text()
    
    Args:
        url: URL of article to summarize
        instructions: Optional user instructions
    
    Returns:
        Dictionary containing:
            - success: Whether extraction succeeded
            - summary: Generated summary
            - url: Original URL
            - title: Article title (if available)
            - author: Article author (if available)
            - date: Publication date (if available)
            - num_chunks: Number of chunks
            - error: Error message (if failed)
    
    Example:
        >>> result = summarize_url("https://example.com/news/article")
        >>> if result["success"]:
        ...     print(result["summary"])
    """
    pipeline = get_pipeline()
    return pipeline.summarize_url(url, instructions)


def summarize_urls(
    urls: List[str],
    instructions: Optional[str] = None,
    combine: bool = True
) -> Dict:
    """
    Summarize multiple URLs.
    
    Args:
        urls: List of URLs to summarize
        instructions: Optional user instructions
        combine: If True, combines all articles into single summary.
                If False, returns separate summary for each URL.
    
    Returns:
        Dictionary containing:
            - success: Whether any extraction succeeded
            - summary: Combined summary (if combine=True)
            - summaries: List of individual summaries (if combine=False)
            - num_urls: Total number of URLs
            - successful_extractions: Number of successful extractions
            - url_results: Detailed results for each URL
    
    Example:
        >>> urls = ["https://site1.com/article1", "https://site2.com/article2"]
        >>> result = summarize_urls(urls, instructions="compare key themes")
        >>> print(result["summary"])
    """
    pipeline = get_pipeline()
    return pipeline.summarize_urls(urls, instructions, combine)


def summarize_file(
    filepath: str,
    instructions: Optional[str] = None
) -> Dict:
    """
    Summarize from file (PDF, DOCX, or TXT).
    
    Supported formats:
    - PDF (.pdf): Extracted using pdfplumber
    - DOCX (.docx, .doc): Extracted using python-docx
    - TXT (.txt): Read directly
    
    Args:
        filepath: Path to file
        instructions: Optional user instructions
    
    Returns:
        Dictionary containing:
            - success: Whether extraction succeeded
            - summary: Generated summary
            - file_path: Original file path
            - file_type: File extension
            - title: Document title (if available)
            - author: Document author (if available)
            - num_pages: Number of pages (for PDF)
            - error: Error message (if failed)
    
    Example:
        >>> result = summarize_file("report.pdf", instructions="focus on key findings")
        >>> if result["success"]:
        ...     print(result["summary"])
    """
    pipeline = get_pipeline()
    return pipeline.summarize_file(filepath, instructions)


def summarize_files(
    filepaths: List[str],
    instructions: Optional[str] = None,
    combine: bool = True
) -> Dict:
    """
    Summarize multiple files (PDF, DOCX, or TXT).
    
    Args:
        filepaths: List of file paths
        instructions: Optional user instructions
        combine: If True, combines all files into single summary.
                If False, returns separate summary for each file.
    
    Returns:
        Dictionary containing:
            - success: Whether any extraction succeeded
            - summary: Combined summary (if combine=True)
            - summaries: List of individual summaries (if combine=False)
            - num_files: Total number of files
            - successful_extractions: Number of successful extractions
            - file_results: Detailed results for each file
    
    Example:
        >>> files = ["report1.pdf", "report2.pdf", "notes.txt"]
        >>> result = summarize_files(files, instructions="brief", combine=True)
        >>> print(result["summary"])
    """
    pipeline = get_pipeline()
    return pipeline.summarize_files(filepaths, instructions, combine)



def refine_summary(
    summary: str,
    feedback: str,
    original_text: Optional[str] = None
) -> Dict:
    """
    Refine existing summary based on user feedback.
    
    This function uses prompt-based refinement (no retraining).
    It parses the feedback and regenerates the summary with
    the specified modifications.
    
    Common feedback patterns:
    - "make it shorter" / "make it longer"
    - "format as bullet points"
    - "create a headline"
    - "include numbers and statistics"
    - "use simple language"
    - "add more technical details"
    
    Args:
        summary: Original summary to refine
        feedback: User feedback/instructions
        original_text: Optional original text for better context
    
    Returns:
        Dictionary containing:
            - refined_summary: New refined summary
            - original_summary: Original summary
            - feedback: User feedback
            - parsed_instruction: Parsed instruction details
    
    Example:
        >>> original = "AI is advancing rapidly."
        >>> refined = refine_summary(original, "make it more detailed")
        >>> print(refined["refined_summary"])
    """
    pipeline = get_pipeline()
    return pipeline.refine_summary(summary, feedback, original_text)


def get_model_info() -> Dict:
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary with model information:
            - model_type: Model architecture
            - model_name: Model name
            - device: Device (CPU/GPU/MPS)
            - max_length: Maximum sequence length
            - vocab_size: Vocabulary size
            - loaded: Whether model is loaded
    
    Example:
        >>> info = get_model_info()
        >>> print(f"Model: {info['model_name']} on {info['device']}")
    """
    from summarization.model_loader import get_model_info
    return get_model_info()


# Convenience aliases
summarize = summarize_text
summarize_document = summarize_file


if __name__ == "__main__":
    # Example usage
    print("RLHF News Summarization System - Utils")
    print("=" * 50)
    
    # Test text summarization
    test_text = """
    Artificial intelligence has made remarkable progress in recent years.
    Machine learning algorithms can now perform tasks that were once thought
    to require human intelligence. Deep learning, a subset of machine learning,
    has been particularly successful in areas like computer vision and natural
    language processing. These advances are transforming industries from
    healthcare to finance to transportation.
    """
    
    print("\n1. Testing text summarization...")
    result = summarize_text(test_text)
    print(f"Summary: {result['summary']}")
    print(f"Chunks: {result['num_chunks']}")
    
    print("\n2. Testing refinement...")
    refined = refine_summary(result['summary'], "make it shorter")
    print(f"Refined: {refined['refined_summary']}")
    
    print("\n3. Model info...")
    info = get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Device: {info['device']}")
    
    print("\nAll tests completed!")
