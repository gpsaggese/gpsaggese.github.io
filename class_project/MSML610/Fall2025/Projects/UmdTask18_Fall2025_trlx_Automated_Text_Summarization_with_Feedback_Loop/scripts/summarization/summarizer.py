"""
Core Summarization Engine

Handles text summarization using the DPO-trained T5-large model.
Supports different output styles and instruction-based refinement.
"""

import torch
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Summarizer:
    """Core summarization engine using DPO T5-large."""
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize summarizer.
        
        Args:
            model: T5 model instance
            tokenizer: T5 tokenizer instance
            device: Device (CPU/GPU/MPS)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def summarize(
        self,
        text: str,
        max_length: int = 400,
        min_length: int = 150,
        num_beams: int = 4,
        length_penalty: float = 1.5,
        early_stopping: bool = True,
        instructions: Optional[str] = None
    ) -> str:
        """
        Generate summary for input text.
        
        Args:
            text: Input text (should start with 'summarize: ')
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for generation
            early_stopping: Whether to stop early
            instructions: Optional user instructions for refinement
        
        Returns:
            Generated summary
        """
        # Optimized parameters to prevent repetition and ensure quality
        max_length = 400  # Optimal for complete thoughts
        min_length = 150  # Natural stopping point without forcing padding
        length_penalty = 2.0  
        no_repeat_ngram_size = 4  
        repetition_penalty = 1.5    
        # Pass user instructions directly to the model
        if instructions:
            # Remove existing prefix if present
            clean_text = text.replace("summarize: ", "")
            text = f"summarize: {clean_text}. {instructions}"
        
        # Ensure text has prefix
        if not text.startswith("summarize"):
            text = f"summarize: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with anti-repetition parameters
        logger.info(f"Generating with max_length={max_length}, min_length={min_length}, repetition_penalty={repetition_penalty}")
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                num_beams=4,
                early_stopping=True,  # Stop when quality degrades
                do_sample=False  # Use deterministic beam search
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process: Ensure summary ends with complete sentence
        summary = self._ensure_complete_sentence(summary)
        
        return summary
    
    def _ensure_complete_sentence(self, text: str) -> str:
        """
        Ensure summary ends with a complete sentence.
        Removes any incomplete sentence at the end.
        
        Args:
            text: Generated summary text
        
        Returns:
            Text with complete sentences only
        """
        import re
        
        # Remove trailing whitespace
        text = text.strip()
        
        # Check if text ends with sentence-ending punctuation
        if text and text[-1] not in '.!?':
            # Find the last sentence-ending punctuation
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > 0:
                # Truncate to last complete sentence
                text = text[:last_period + 1].strip()
        
        return text
    
    def _format_into_paragraphs(self, text: str, instructions: str) -> str:
        """
        Format summary into paragraphs.
        
        Args:
            text: Generated summary text
            instructions: Original instructions
        
        Returns:
            Formatted text with paragraph breaks
        """
        import re
        
        # Extract number of paragraphs requested
        match = re.search(r'(\d+)\s*paragraph', instructions.lower())
        if match:
            num_paragraphs = int(match.group(1))
        else:
            num_paragraphs = 3  # Default
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        if len(sentences) <= num_paragraphs:
            # Not enough sentences, return as is
            return text
        
        # Distribute sentences across paragraphs
        sentences_per_para = len(sentences) // num_paragraphs
        remainder = len(sentences) % num_paragraphs
        
        paragraphs = []
        idx = 0
        
        for i in range(num_paragraphs):
            # Add extra sentence to first paragraphs if there's remainder
            para_size = sentences_per_para + (1 if i < remainder else 0)
            para_sentences = sentences[idx:idx + para_size]
            paragraphs.append(' '.join(para_sentences))
            idx += para_size
        
        # Join with double newlines for paragraph breaks
        return '\n\n'.join(paragraphs)
    
    def summarize_batch(
        self,
        texts: List[str],
        max_length: int = 450,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Summarize multiple texts in batches.
        
        Args:
            texts: List of input texts
            max_length: Maximum summary length
            batch_size: Batch size for processing
            **kwargs: Additional arguments for summarize()
        
        Returns:
            List of summaries
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                summary = self.summarize(text, max_length=max_length, **kwargs)
                summaries.append(summary)
        
        return summaries
    
    def summarize_with_style(
        self,
        text: str,
        style: str = "default",
        **kwargs
    ) -> str:
        """
        Summarize with specific output style.
        
        Args:
            text: Input text
            style: Output style ('default', 'bullet_points', 'headline', 'brief')
            **kwargs: Additional arguments for summarize()
        
        Returns:
            Styled summary
        """
        style_instructions = {
            "default": None,
            "bullet_points": "Format as bullet points",
            "headline": "Create a single headline",
            "brief": "Make it very brief, one sentence only",
            "detailed": "Provide a detailed summary with key points",
            "technical": "Use technical language and include specific details",
            "simple": "Use simple language suitable for general audience"
        }
        
        instructions = style_instructions.get(style, None)
        
        if instructions:
            return self.summarize(text, instructions=instructions, **kwargs)
        else:
            return self.summarize(text, **kwargs)
    
    def refine_summary(
        self,
        original_summary: str,
        feedback: str,
        original_text: Optional[str] = None
    ) -> str:
        """
        Refine an existing summary based on user feedback.
        
        Args:
            original_summary: The original summary to refine
            feedback: User feedback/instructions
            original_text: Optional original text for context
        
        Returns:
            Refined summary
        """
        # Build refinement prompt
        if original_text:
            # Re-summarize with feedback
            prompt = f"summarize: {original_text}. Instructions: {feedback}"
        else:
            # Refine existing summary
            prompt = f"summarize: {original_summary}. Instructions: {feedback}"
        
        return self.summarize(prompt)


def create_summarizer(model=None, tokenizer=None, device=None):
    """
    Create a summarizer instance.
    
    Args:
        model: T5 model (if None, loads from model_loader)
        tokenizer: T5 tokenizer (if None, loads from model_loader)
        device: Device (if None, loads from model_loader)
    
    Returns:
        Summarizer instance
    """
    if model is None or tokenizer is None or device is None:
        from .model_loader import load_summarization_model
        model, tokenizer, device = load_summarization_model()
    
    return Summarizer(model, tokenizer, device)


if __name__ == "__main__":
    # Test summarizer
    from model_loader import load_summarization_model
    
    print("Loading model...")
    model, tokenizer, device = load_summarization_model()
    
    print("Creating summarizer...")
    summarizer = Summarizer(model, tokenizer, device)
    
    # Test basic summarization
    test_text = """
    Artificial intelligence has made remarkable progress in recent years.
    Machine learning algorithms can now perform tasks that were once thought
    to require human intelligence. Deep learning, a subset of machine learning,
    has been particularly successful in areas like computer vision and natural
    language processing. These advances are transforming industries from
    healthcare to finance to transportation.
    """
    
    print("\nTest 1: Basic summarization")
    summary = summarizer.summarize(test_text)
    print(f"Summary: {summary}")
    
    print("\nTest 2: Bullet points style")
    summary = summarizer.summarize_with_style(test_text, style="bullet_points")
    print(f"Summary: {summary}")
    
    print("\nTest 3: Headline style")
    summary = summarizer.summarize_with_style(test_text, style="headline")
    print(f"Summary: {summary}")
    
    print("\nTest 4: Refinement")
    original = "AI has made progress in recent years."
    refined = summarizer.refine_summary(original, "make it more detailed")
    print(f"Refined: {refined}")
