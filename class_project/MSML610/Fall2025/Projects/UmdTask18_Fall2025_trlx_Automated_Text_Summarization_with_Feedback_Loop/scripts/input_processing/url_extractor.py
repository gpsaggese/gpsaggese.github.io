"""
URL Article Extractor

Fetches and extracts clean article text from URLs using trafilatura.
Removes boilerplate content like ads, menus, and navigation.
"""

import requests
from typing import Optional, Dict
import logging

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logging.warning("trafilatura not installed. Install with: pip install trafilatura")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URLExtractor:
    """Extract article text from URLs."""
    
    def __init__(self, timeout=30):
        """
        Initialize URL extractor.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        
        if not TRAFILATURA_AVAILABLE:
            raise ImportError(
                "trafilatura is required for URL extraction. "
                "Install with: pip install trafilatura"
            )
    
    def extract_from_url(self, url: str) -> Optional[str]:
        """
        Extract article text from URL.
        
        Args:
            url: URL to extract from
        
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            logger.info(f"Fetching URL: {url}")
            
            # Fetch HTML
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
            
            # Extract article text
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            
            if text:
                logger.info(f"Successfully extracted {len(text)} characters")
                return text
            else:
                logger.warning(f"No text extracted from {url}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting from URL {url}: {e}")
            return None
    
    def extract_with_metadata(self, url: str) -> Dict:
        """
        Extract article with metadata.
        
        Args:
            url: URL to extract from
        
        Returns:
            Dictionary with text and metadata
        """
        try:
            # Fetch HTML
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
            
            # Extract with metadata
            metadata = trafilatura.extract_metadata(html)
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            
            result = {
                "url": url,
                "text": text,
                "title": metadata.title if metadata else None,
                "author": metadata.author if metadata else None,
                "date": metadata.date if metadata else None,
                "description": metadata.description if metadata else None,
                "success": text is not None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from URL {url}: {e}")
            return {
                "url": url,
                "text": None,
                "title": None,
                "author": None,
                "date": None,
                "description": None,
                "success": False,
                "error": str(e)
            }
    
    def extract_from_multiple_urls(self, urls: list) -> Dict[str, str]:
        """
        Extract from multiple URLs.
        
        Args:
            urls: List of URLs
        
        Returns:
            Dictionary mapping URLs to extracted text
        """
        results = {}
        
        for url in urls:
            text = self.extract_from_url(url)
            if text:
                results[url] = text
        
        logger.info(f"Successfully extracted from {len(results)}/{len(urls)} URLs")
        
        return results


def extract_from_url(url: str, timeout=30) -> Optional[str]:
    """
    Convenience function to extract from URL.
    
    Args:
        url: URL to extract from
        timeout: Request timeout
    
    Returns:
        Extracted text or None
    """
    extractor = URLExtractor(timeout=timeout)
    return extractor.extract_from_url(url)


if __name__ == "__main__":
    # Test URL extraction
    test_urls = [
        "https://www.bbc.com/news/technology",
        "https://techcrunch.com"
    ]
    
    extractor = URLExtractor()
    
    for url in test_urls:
        print(f"\nTesting URL: {url}")
        result = extractor.extract_with_metadata(url)
        
        if result["success"]:
            print(f"Title: {result['title']}")
            print(f"Author: {result['author']}")
            print(f"Date: {result['date']}")
            print(f"Text preview: {result['text'][:200]}...")
        else:
            print(f"Extraction failed: {result.get('error', 'Unknown error')}")
