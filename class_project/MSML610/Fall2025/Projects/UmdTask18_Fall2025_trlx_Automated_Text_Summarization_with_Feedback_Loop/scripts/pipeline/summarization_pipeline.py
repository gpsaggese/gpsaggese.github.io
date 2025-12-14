"""
Summarization Pipeline

End-to-end orchestration of the summarization workflow.
Handles multiple input formats, chunking, summarization, and aggregation.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional, List, Dict, Union
import logging

from input_processing.url_extractor import URLExtractor
from input_processing.pdf_extractor import PDFExtractor
from input_processing.docx_extractor import DOCXExtractor
from input_processing.text_cleaner import TextCleaner

from summarization.model_loader import SummarizationModel
from summarization.chunker import TextChunker
from summarization.summarizer import Summarizer
from summarization.aggregator import HierarchicalAggregator

from refinement.instruction_parser import InstructionParser
from refinement.prompt_builder import PromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationPipeline:
    """End-to-end summarization pipeline."""
    
    def __init__(self, model_path=None):
        """
        Initialize pipeline.
        
        Args:
            model_path: Optional path to model checkpoint
        """
        # Initialize model
        logger.info("Initializing summarization pipeline...")
        model_loader = SummarizationModel()
        self.model, self.tokenizer, self.device = model_loader.load_model(model_path)
        
        # Initialize components
        self.summarizer = Summarizer(self.model, self.tokenizer, self.device)
        self.chunker = TextChunker(self.tokenizer, max_tokens=900, overlap_tokens=100)
        self.aggregator = HierarchicalAggregator(self.summarizer)
        
        self.text_cleaner = TextCleaner()
        self.instruction_parser = InstructionParser()
        self.prompt_builder = PromptBuilder()
        
        # Input extractors (lazy initialization)
        self._url_extractor = None
        self._pdf_extractor = None
        self._docx_extractor = None
        
        logger.info("Pipeline initialized successfully")
    
    @property
    def url_extractor(self):
        """Lazy load URL extractor."""
        if self._url_extractor is None:
            self._url_extractor = URLExtractor()
        return self._url_extractor
    
    @property
    def pdf_extractor(self):
        """Lazy load PDF extractor."""
        if self._pdf_extractor is None:
            self._pdf_extractor = PDFExtractor()
        return self._pdf_extractor
    
    @property
    def docx_extractor(self):
        """Lazy load DOCX extractor."""
        if self._docx_extractor is None:
            self._docx_extractor = DOCXExtractor()
        return self._docx_extractor
    
    def summarize_text(
        self,
        text: str,
        instructions: Optional[str] = None,
        clean_text: bool = True
    ) -> Dict:
        """
        Summarize raw text.
        
        Args:
            text: Input text
            instructions: Optional user instructions
            clean_text: Whether to clean text first
        
        Returns:
            Dictionary with summary and metadata
        """
        logger.info("Summarizing text...")
        
        # Clean text
        if clean_text:
            text = self.text_cleaner.clean(text)
        
        # Chunk text
        chunks = self.chunker.chunk_text(text, add_prefix=True)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Summarize chunks
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            summary = self.summarizer.summarize(chunk, instructions=instructions)
            chunk_summaries.append(summary)
        
        # Aggregate if multiple chunks
        if len(chunk_summaries) > 1:
            final_summary = self.aggregator.aggregate(chunk_summaries, instructions)
        else:
            final_summary = chunk_summaries[0] if chunk_summaries else ""
        
        return {
            "summary": final_summary,
            "num_chunks": len(chunks),
            "chunk_summaries": chunk_summaries,
            "input_length": len(text),
            "summary_length": len(final_summary)
        }
    
    def summarize_url(
        self,
        url: str,
        instructions: Optional[str] = None
    ) -> Dict:
        """
        Summarize article from URL.
        
        Args:
            url: URL to summarize
            instructions: Optional user instructions
        
        Returns:
            Dictionary with summary and metadata
        """
        logger.info(f"Summarizing URL: {url}")
        
        # Extract text from URL
        result = self.url_extractor.extract_with_metadata(url)
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Failed to extract from URL"),
                "url": url
            }
        
        # Summarize extracted text
        summary_result = self.summarize_text(result["text"], instructions)
        
        # Add URL metadata
        summary_result.update({
            "success": True,
            "url": url,
            "title": result.get("title"),
            "author": result.get("author"),
            "date": result.get("date")
        })
        
        return summary_result
    
    def summarize_urls(
        self,
        urls: List[str],
        instructions: Optional[str] = None,
        combine: bool = True
    ) -> Dict:
        """
        Summarize multiple URLs.
        
        Args:
            urls: List of URLs
            instructions: Optional user instructions
            combine: Whether to combine into single summary
        
        Returns:
            Dictionary with summaries and metadata
        """
        logger.info(f"Summarizing {len(urls)} URLs...")
        
        # Extract from all URLs
        url_results = []
        all_texts = []
        
        for url in urls:
            result = self.url_extractor.extract_with_metadata(url)
            url_results.append(result)
            if result["success"]:
                all_texts.append(result["text"])
        
        if not all_texts:
            return {
                "success": False,
                "error": "No text extracted from any URL",
                "urls": urls
            }
        
        if combine:
            # Combine all texts and summarize
            combined_text = "\n\n".join(all_texts)
            summary_result = self.summarize_text(combined_text, instructions)
            
            return {
                "success": True,
                "summary": summary_result["summary"],
                "num_urls": len(urls),
                "successful_extractions": len(all_texts),
                "url_results": url_results,
                **summary_result
            }
        else:
            # Summarize each URL separately
            summaries = []
            for text in all_texts:
                result = self.summarize_text(text, instructions)
                summaries.append(result["summary"])
            
            return {
                "success": True,
                "summaries": summaries,
                "num_urls": len(urls),
                "successful_extractions": len(all_texts),
                "url_results": url_results
            }
    
    def summarize_file(
        self,
        filepath: str,
        instructions: Optional[str] = None
    ) -> Dict:
        """
        Summarize from file (PDF, DOCX, or TXT).
        
        Args:
            filepath: Path to file
            instructions: Optional user instructions
        
        Returns:
            Dictionary with summary and metadata
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {
                "success": False,
                "error": f"File not found: {filepath}"
            }
        
        logger.info(f"Summarizing file: {filepath}")
        
        # Determine file type and extract
        suffix = filepath.suffix.lower()
        
        if suffix == ".pdf":
            result = self.pdf_extractor.extract_with_metadata(str(filepath))
        elif suffix in [".docx", ".doc"]:
            result = self.docx_extractor.extract_with_metadata(str(filepath))
        elif suffix == ".txt":
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            result = {
                "file_path": str(filepath),
                "text": text,
                "success": True
            }
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {suffix}"
            }
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Failed to extract from file"),
                "file_path": str(filepath)
            }
        
        # Summarize extracted text
        summary_result = self.summarize_text(result["text"], instructions)
        
        # Add file metadata
        summary_result.update({
            "success": True,
            "file_path": str(filepath),
            "file_type": suffix,
            **{k: v for k, v in result.items() if k not in ["text", "success"]}
        })
        
        return summary_result
    
    def summarize_files(
        self,
        filepaths: List[str],
        instructions: Optional[str] = None,
        combine: bool = True
    ) -> Dict:
        """
        Summarize multiple files.
        
        Args:
            filepaths: List of file paths
            instructions: Optional user instructions
            combine: Whether to combine into single summary
        
        Returns:
            Dictionary with summaries and metadata
        """
        logger.info(f"Summarizing {len(filepaths)} files...")
        
        # Extract text from all files first (without summarizing yet)
        file_results = []
        all_texts = []
        
        for filepath in filepaths:
            filepath = Path(filepath)
            
            if not filepath.exists():
                file_results.append({"success": False, "error": f"File not found: {filepath}"})
                continue
            
            # Determine file type and extract text
            suffix = filepath.suffix.lower()
            
            if suffix == ".pdf":
                result = self.pdf_extractor.extract_with_metadata(str(filepath))
            elif suffix in [".docx", ".doc"]:
                result = self.docx_extractor.extract_with_metadata(str(filepath))
            elif suffix == ".txt":
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                result = {
                    "file_path": str(filepath),
                    "text": text,
                    "success": True
                }
            else:
                result = {"success": False, "error": f"Unsupported file type: {suffix}"}
            
            file_results.append(result)
            if result.get("success"):
                extracted_text = result.get("text", "")
                all_texts.append(extracted_text)
                logger.info(f"Extracted {len(extracted_text)} characters from {filepath.name}")
        
        if not all_texts:
            return {
                "success": False,
                "error": "No text extracted from any file",
                "filepaths": filepaths
            }
        
        # Log combined text length
        combined_text = "\n\n".join(all_texts)
        logger.info(f"Combined text from {len(all_texts)} files: {len(combined_text)} total characters")
        
        if combine:
            # Combine all texts and summarize once
            combined_text = "\n\n".join(all_texts)
            summary_result = self.summarize_text(combined_text, instructions)
            
            return {
                "success": True,
                "summary": summary_result["summary"],
                "num_files": len(filepaths),
                "successful_extractions": len(all_texts),
                "file_results": file_results,
                **summary_result
            }
        else:
            # Summarize each text separately
            summaries = []
            for text in all_texts:
                result = self.summarize_text(text, instructions)
                summaries.append(result["summary"])
            
            return {
                "success": True,
                "summaries": summaries,
                "num_files": len(filepaths),
                "successful_extractions": len(all_texts),
                "file_results": file_results
            }
    
    
    def refine_summary(
        self,
        summary: str,
        feedback: str,
        original_text: Optional[str] = None
    ) -> Dict:
        """
        Refine existing summary based on feedback.
        
        Args:
            summary: Original summary
            feedback: User feedback
            original_text: Optional original text for context
        
        Returns:
            Dictionary with refined summary
        """
        logger.info("Refining summary...")
        
        # Parse feedback
        parsed = self.instruction_parser.parse(feedback)
        formatted_instruction = parsed["formatted_instruction"]
        
        # Refine summary
        refined = self.summarizer.refine_summary(
            summary,
            formatted_instruction or feedback,
            original_text
        )
        
        return {
            "refined_summary": refined,
            "original_summary": summary,
            "feedback": feedback,
            "parsed_instruction": parsed
        }


# Singleton instance
_pipeline_instance = None


def get_pipeline(model_path=None):
    """Get or create pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SummarizationPipeline(model_path)
    return _pipeline_instance


def reset_pipeline():
    """Reset pipeline singleton - forces recreation with new parameters."""
    global _pipeline_instance
    _pipeline_instance = None
    logger.info("Pipeline singleton reset")


if __name__ == "__main__":
    # Test pipeline
    pipeline = SummarizationPipeline()
    
    test_text = """
    Artificial intelligence has made remarkable progress in recent years.
    Machine learning algorithms can now perform tasks that were once thought
    to require human intelligence. Deep learning, a subset of machine learning,
    has been particularly successful in areas like computer vision and natural
    language processing. These advances are transforming industries from
    healthcare to finance to transportation. However, challenges remain in
    areas like explainability, fairness, and safety.
    """
    
    print("Testing text summarization...")
    result = pipeline.summarize_text(test_text)
    print(f"Summary: {result['summary']}")
    print(f"Chunks: {result['num_chunks']}")
