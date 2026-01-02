"""
Example Usage: Modular News Summarization System

This script demonstrates various use cases of the summarization system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils import (
    summarize_text,
    summarize_url,
    summarize_urls,
    summarize_file,
    refine_summary,
    get_model_info
)


def example_1_text_summarization():
    """Example 1: Basic text summarization."""
    print("\n" + "="*80)
    print("Example 1: Text Summarization")
    print("="*80)
    
    text = """
    Artificial intelligence has made remarkable progress in recent years, with
    breakthroughs in machine learning enabling computers to perform tasks that
    once required human intelligence. Deep learning, a subset of machine learning,
    has been particularly successful in areas like computer vision, natural language
    processing, and speech recognition. These advances are transforming industries
    from healthcare to finance to transportation. However, challenges remain in
    areas like explainability, fairness, and safety. Researchers are working to
    develop AI systems that are not only powerful but also trustworthy and aligned
    with human values.
    """
    
    result = summarize_text(text)
    
    print(f"\nOriginal length: {result['input_length']} characters")
    print(f"Number of chunks: {result['num_chunks']}")
    print(f"\nSummary: {result['summary']}")
    print(f"Summary length: {result['summary_length']} characters")


def example_2_url_summarization():
    """Example 2: Summarize from URL."""
    print("\n" + "="*80)
    print("Example 2: URL Summarization")
    print("="*80)
    
    # Example URL (replace with actual news URL)
    url = "https://www.bbc.com/news/technology"
    
    print(f"\nSummarizing URL: {url}")
    result = summarize_url(url)
    
    if result["success"]:
        print(f"\nTitle: {result.get('title', 'N/A')}")
        print(f"Author: {result.get('author', 'N/A')}")
        print(f"Date: {result.get('date', 'N/A')}")
        print(f"\nSummary: {result['summary']}")
    else:
        print(f"\nFailed to extract from URL: {result.get('error')}")


def example_3_multi_url_summarization():
    """Example 3: Summarize multiple URLs."""
    print("\n" + "="*80)
    print("Example 3: Multi-URL Summarization")
    print("="*80)
    
    urls = [
        "https://techcrunch.com",
        "https://www.theverge.com"
    ]
    
    print(f"\nSummarizing {len(urls)} URLs...")
    result = summarize_urls(urls, instructions="compare key themes", combine=True)
    
    if result["success"]:
        print(f"\nSuccessfully extracted from {result['successful_extractions']}/{result['num_urls']} URLs")
        print(f"\nCombined Summary: {result['summary']}")
    else:
        print(f"\nFailed: {result.get('error')}")


def example_4_file_summarization():
    """Example 4: Summarize from file."""
    print("\n" + "="*80)
    print("Example 4: File Summarization")
    print("="*80)
    
    # Create a sample text file
    sample_file = Path(__file__).parent / "sample_article.txt"
    
    with open(sample_file, 'w') as f:
        f.write("""
        Climate change is one of the most pressing challenges facing humanity.
        Rising global temperatures are causing more frequent and severe weather
        events, including hurricanes, droughts, and floods. Scientists warn that
        without immediate action to reduce greenhouse gas emissions, the impacts
        will become increasingly severe. Governments and organizations worldwide
        are working to transition to renewable energy sources and implement
        sustainable practices. However, the pace of change needs to accelerate
        to meet international climate goals.
        """)
    
    print(f"\nSummarizing file: {sample_file}")
    result = summarize_file(str(sample_file))
    
    if result["success"]:
        print(f"\nFile type: {result['file_type']}")
        print(f"Summary: {result['summary']}")
    else:
        print(f"\nFailed: {result.get('error')}")
    
    # Clean up
    sample_file.unlink()


def example_5_refinement():
    """Example 5: Iterative refinement."""
    print("\n" + "="*80)
    print("Example 5: Summary Refinement")
    print("="*80)
    
    text = """
    The stock market experienced significant volatility today, with the Dow Jones
    Industrial Average dropping 500 points in early trading before recovering
    slightly by the close. Technology stocks led the decline, with major companies
    like Apple, Microsoft, and Google all posting losses. Analysts attribute the
    sell-off to concerns about rising interest rates and inflation. The Federal
    Reserve is expected to announce its decision on rates next week, which could
    further impact market sentiment.
    """
    
    # Initial summary
    print("\nGenerating initial summary...")
    result = summarize_text(text)
    original = result['summary']
    print(f"Original summary: {original}")
    
    # Refinement 1: Make it shorter
    print("\n--- Refinement 1: Make it shorter ---")
    refined1 = refine_summary(original, "make it shorter, one sentence only")
    print(f"Refined: {refined1['refined_summary']}")
    
    # Refinement 2: Add bullet points
    print("\n--- Refinement 2: Format as bullet points ---")
    refined2 = refine_summary(original, "format as bullet points")
    print(f"Refined: {refined2['refined_summary']}")
    
    # Refinement 3: Include numbers
    print("\n--- Refinement 3: Include specific numbers ---")
    refined3 = refine_summary(original, "include all specific numbers and statistics")
    print(f"Refined: {refined3['refined_summary']}")


def example_6_custom_instructions():
    """Example 6: Custom instructions."""
    print("\n" + "="*80)
    print("Example 6: Custom Instructions")
    print("="*80)
    
    text = """
    A new study published in Nature reveals that regular exercise can significantly
    reduce the risk of cardiovascular disease. Researchers followed 10,000 participants
    over 15 years and found that those who exercised at least 30 minutes per day had
    a 40% lower risk of heart disease compared to sedentary individuals. The study
    also found benefits for mental health, with exercisers reporting lower levels of
    stress and anxiety. Dr. Sarah Johnson, lead author of the study, recommends
    incorporating both aerobic and strength training exercises for optimal health benefits.
    """
    
    instructions = [
        "create a headline",
        "make it very brief",
        "format as bullet points with key findings",
        "use simple language for general audience",
        "focus on the numbers and statistics"
    ]
    
    for instruction in instructions:
        print(f"\n--- Instruction: {instruction} ---")
        result = summarize_text(text, instructions=instruction)
        print(f"Summary: {result['summary']}")


def example_7_model_info():
    """Example 7: Get model information."""
    print("\n" + "="*80)
    print("Example 7: Model Information")
    print("="*80)
    
    info = get_model_info()
    
    print(f"\nModel Type: {info['model_type']}")
    print(f"Model Name: {info['model_name']}")
    print(f"Device: {info['device']}")
    print(f"Max Length: {info['max_length']}")
    print(f"Vocab Size: {info['vocab_size']}")
    print(f"Loaded: {info['loaded']}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("RLHF News Summarization System - Example Usage")
    print("="*80)
    
    try:
        # Run examples
        example_1_text_summarization()
        example_5_refinement()
        example_6_custom_instructions()
        example_7_model_info()
        
        # Optional: Uncomment to test URL/file examples
        # example_2_url_summarization()
        # example_3_multi_url_summarization()
        # example_4_file_summarization()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
