"""
Text Cleaner

Normalizes and cleans extracted text from various sources.
Handles encoding issues, whitespace, and formatting artifacts.
"""

import re
import unicodedata
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and normalize text."""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
        
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Remove trailing/leading whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def fix_encoding(text: str) -> str:
        """
        Fix common encoding issues.
        
        Args:
            text: Input text
        
        Returns:
            Text with fixed encoding
        """
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Fix common replacements
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Ellipsis
            '\xa0': ' ',    # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text
        
        Returns:
            Text without URLs
        """
        # Remove http/https URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove www URLs
        text = re.sub(r'www\.\S+', '', text)
        
        return text
    
    @staticmethod
    def remove_special_characters(text: str, keep_punctuation=True) -> str:
        """
        Remove special characters.
        
        Args:
            text: Input text
            keep_punctuation: Whether to keep basic punctuation
        
        Returns:
            Cleaned text
        """
        if keep_punctuation:
            # Keep letters, numbers, and basic punctuation
            text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
        else:
            # Keep only letters, numbers, and spaces
            text = re.sub(r'[^\w\s]+', '', text)
        
        return text
    
    @staticmethod
    def fix_line_breaks(text: str) -> str:
        """
        Fix awkward line breaks (common in PDFs).
        
        Args:
            text: Input text
        
        Returns:
            Text with fixed line breaks
        """
        # Join lines that don't end with sentence-ending punctuation
        lines = text.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                fixed_lines.append('')
                continue
            
            # If line doesn't end with punctuation and next line exists,
            # join with next line
            if i < len(lines) - 1 and not line[-1] in '.!?:;':
                # Don't add newline, will be joined
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Join consecutive non-empty lines
        result = []
        current_para = []
        
        for line in fixed_lines:
            if line:
                current_para.append(line)
            else:
                if current_para:
                    result.append(' '.join(current_para))
                    current_para = []
                result.append('')
        
        if current_para:
            result.append(' '.join(current_para))
        
        return '\n'.join(result)
    
    def clean(
        self,
        text: str,
        fix_encoding: bool = True,
        normalize_whitespace: bool = True,
        remove_urls: bool = False,
        fix_line_breaks: bool = True
    ) -> str:
        """
        Clean text with specified options.
        
        Args:
            text: Input text
            fix_encoding: Fix encoding issues
            normalize_whitespace: Normalize whitespace
            remove_urls: Remove URLs
            fix_line_breaks: Fix line breaks
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        if fix_encoding:
            text = self.fix_encoding(text)
        
        if fix_line_breaks:
            text = self.fix_line_breaks(text)
        
        if remove_urls:
            text = self.remove_urls(text)
        
        if normalize_whitespace:
            text = self.normalize_whitespace(text)
        
        return text


def clean_text(
    text: str,
    fix_encoding: bool = True,
    normalize_whitespace: bool = True,
    remove_urls: bool = False,
    fix_line_breaks: bool = True
) -> str:
    """
    Convenience function to clean text.
    
    Args:
        text: Input text
        fix_encoding: Fix encoding issues
        normalize_whitespace: Normalize whitespace
        remove_urls: Remove URLs
        fix_line_breaks: Fix line breaks
    
    Returns:
        Cleaned text
    """
    cleaner = TextCleaner()
    return cleaner.clean(text, fix_encoding, normalize_whitespace, remove_urls, fix_line_breaks)


if __name__ == "__main__":
    # Test text cleaner
    test_text = """
    This   is    a   test   text   with    multiple    spaces.
    
    
    
    It has multiple newlines.
    
    It also has special characters: \u2018quotes\u2019 and \u2014dashes\u2014.
    
    And some URLs: https://example.com and www.test.com
    
    This line doesn't end with punctuation
    But this one does.
    """
    
    cleaner = TextCleaner()
    
    print("Original text:")
    print(repr(test_text))
    
    print("\nCleaned text:")
    cleaned = cleaner.clean(test_text, remove_urls=True)
    print(repr(cleaned))
    
    print("\nFinal output:")
    print(cleaned)
