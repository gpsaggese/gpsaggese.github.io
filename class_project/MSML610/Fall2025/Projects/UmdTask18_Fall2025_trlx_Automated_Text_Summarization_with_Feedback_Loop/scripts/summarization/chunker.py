"""
Text Chunking Module

Implements sentence-aware text chunking that respects token limits
while maintaining coherence and context at chunk boundaries.
"""

import re
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Sentence-aware text chunker for T5 models."""
    
    def __init__(self, tokenizer, max_tokens=900, overlap_tokens=100):
        """
        Initialize text chunker.
        
        Args:
            tokenizer: T5 tokenizer instance
            max_tokens: Maximum tokens per chunk (default: 900, leaves buffer for T5)
            overlap_tokens: Number of overlapping tokens between chunks
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting (handles most cases)
        # Splits on period, exclamation, question mark followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def chunk_text(self, text: str, add_prefix=True) -> List[str]:
        """
        Chunk text into segments respecting token limits.
        
        Args:
            text: Input text to chunk
            add_prefix: Whether to add 'summarize: ' prefix to each chunk
        
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds max_tokens, split it further
            if sentence_tokens > self.max_tokens:
                # If we have accumulated sentences, save them first
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if add_prefix:
                        chunk_text = f"summarize: {chunk_text}"
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word)
                    if temp_tokens + word_tokens > self.max_tokens:
                        if temp_chunk:
                            chunk_text = " ".join(temp_chunk)
                            if add_prefix:
                                chunk_text = f"summarize: {chunk_text}"
                            chunks.append(chunk_text)
                        temp_chunk = [word]
                        temp_tokens = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
            
            # Check if adding this sentence exceeds limit
            elif current_tokens + sentence_tokens > self.max_tokens:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if add_prefix:
                        chunk_text = f"summarize: {chunk_text}"
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.overlap_tokens > 0 and current_chunk:
                    # Keep last few sentences for context
                    overlap_text = " ".join(current_chunk)
                    overlap_actual_tokens = self.count_tokens(overlap_text)
                    
                    if overlap_actual_tokens > self.overlap_tokens:
                        # Trim to overlap size
                        overlap_sentences = []
                        overlap_count = 0
                        for s in reversed(current_chunk):
                            s_tokens = self.count_tokens(s)
                            if overlap_count + s_tokens <= self.overlap_tokens:
                                overlap_sentences.insert(0, s)
                                overlap_count += s_tokens
                            else:
                                break
                        current_chunk = overlap_sentences + [sentence]
                        current_tokens = overlap_count + sentence_tokens
                    else:
                        current_chunk = current_chunk + [sentence]
                        current_tokens = overlap_actual_tokens + sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if add_prefix:
                chunk_text = f"summarize: {chunk_text}"
            chunks.append(chunk_text)
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def get_chunk_info(self, chunks: List[str]) -> List[dict]:
        """
        Get information about chunks.
        
        Args:
            chunks: List of text chunks
        
        Returns:
            List of dictionaries with chunk metadata
        """
        info = []
        for i, chunk in enumerate(chunks):
            # Remove prefix for token counting
            clean_chunk = chunk.replace("summarize: ", "")
            info.append({
                "chunk_id": i,
                "tokens": self.count_tokens(clean_chunk),
                "characters": len(clean_chunk),
                "preview": clean_chunk[:100] + "..." if len(clean_chunk) > 100 else clean_chunk
            })
        return info


def chunk_text(text: str, tokenizer, max_tokens=900, overlap_tokens=100) -> List[str]:
    """
    Convenience function to chunk text.
    
    Args:
        text: Input text
        tokenizer: T5 tokenizer
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunker = TextChunker(tokenizer, max_tokens, overlap_tokens)
    return chunker.chunk_text(text)


if __name__ == "__main__":
    # Test chunking
    from transformers import T5Tokenizer
    
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    chunker = TextChunker(tokenizer, max_tokens=100, overlap_tokens=20)
    
    test_text = """
    Artificial intelligence is transforming the world. Machine learning algorithms
    are becoming more sophisticated. Deep learning has enabled breakthroughs in
    computer vision and natural language processing. Researchers are developing
    new architectures and training methods. The field is advancing rapidly with
    new discoveries every day. Applications span healthcare, finance, transportation,
    and many other domains. Ethical considerations are becoming increasingly important.
    """
    
    chunks = chunker.chunk_text(test_text)
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(chunk)
        print(f"Tokens: {chunker.count_tokens(chunk.replace('summarize: ', ''))}")
