"""
PDF Text Extractor

Extracts and cleans text from PDF files using pdfplumber.
Handles multi-column layouts and formatting artifacts.
"""

from pathlib import Path
from typing import Optional
import logging

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not installed. Install with: pip install pdfplumber")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF files."""
    
    def __init__(self):
        """Initialize PDF extractor."""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install with: pip install pdfplumber"
            )
    
    def extract_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted text or None if extraction fails
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from page
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_parts.append(page_text)
                        logger.debug(f"Extracted {len(page_text)} chars from page {page_num}")
            
            if text_parts:
                full_text = "\n\n".join(text_parts)
                logger.info(f"Successfully extracted {len(full_text)} characters from {len(text_parts)} pages")
                return full_text
            else:
                logger.warning(f"No text extracted from {pdf_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}")
            return None
    
    def extract_with_metadata(self, pdf_path: str) -> dict:
        """
        Extract text with metadata.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with text and metadata
        """
        pdf_path = Path(pdf_path)
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                metadata = pdf.metadata
                
                # Extract text
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                full_text = "\n\n".join(text_parts) if text_parts else None
                
                return {
                    "file_path": str(pdf_path),
                    "text": full_text,
                    "num_pages": len(pdf.pages),
                    "title": metadata.get("Title", None) if metadata else None,
                    "author": metadata.get("Author", None) if metadata else None,
                    "subject": metadata.get("Subject", None) if metadata else None,
                    "creator": metadata.get("Creator", None) if metadata else None,
                    "success": full_text is not None
                }
                
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}")
            return {
                "file_path": str(pdf_path),
                "text": None,
                "num_pages": 0,
                "title": None,
                "author": None,
                "subject": None,
                "creator": None,
                "success": False,
                "error": str(e)
            }


def extract_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Convenience function to extract from PDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text or None
    """
    extractor = PDFExtractor()
    return extractor.extract_from_pdf(pdf_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        
        extractor = PDFExtractor()
        result = extractor.extract_with_metadata(pdf_file)
        
        if result["success"]:
            print(f"File: {result['file_path']}")
            print(f"Pages: {result['num_pages']}")
            print(f"Title: {result['title']}")
            print(f"Author: {result['author']}")
            print(f"\nText preview (first 500 chars):")
            print(result['text'][:500])
            print(f"\n... ({len(result['text'])} total characters)")
        else:
            print(f"Extraction failed: {result.get('error', 'Unknown error')}")
    else:
        print("Usage: python pdf_extractor.py <pdf_file>")
