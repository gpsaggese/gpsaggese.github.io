"""
Hierarchical Aggregation Module

Combines chunk-level summaries into a coherent global summary
using recursive summarization when needed.
"""

from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalAggregator:
    """Aggregates multiple summaries hierarchically."""
    
    def __init__(self, summarizer, max_chunks_per_level=5):
        """
        Initialize aggregator.
        
        Args:
            summarizer: Summarizer instance
            max_chunks_per_level: Maximum chunks before recursive summarization
        """
        self.summarizer = summarizer
        self.max_chunks_per_level = max_chunks_per_level
    
    def aggregate(
        self,
        summaries: List[str],
        instructions: str = None
    ) -> str:
        """
        Aggregate multiple summaries into one.
        
        Args:
            summaries: List of chunk-level summaries
            instructions: Optional user instructions
        
        Returns:
            Final aggregated summary
        """
        if not summaries:
            return ""
        
        if len(summaries) == 1:
            return summaries[0]
        
        logger.info(f"Aggregating {len(summaries)} summaries")
        
        # If summaries fit in one pass, combine directly
        if len(summaries) <= self.max_chunks_per_level:
            return self._combine_summaries(summaries, instructions)
        
        # Otherwise, use recursive aggregation
        return self._recursive_aggregate(summaries, instructions)
    
    def _combine_summaries(
        self,
        summaries: List[str],
        instructions: str = None
    ) -> str:
        """
        Combine summaries in a single pass.
        
        Args:
            summaries: List of summaries to combine
            instructions: Optional instructions
        
        Returns:
            Combined summary
        """
        # Join summaries with clear separation
        combined_text = " ".join(summaries)
        
        # Create prompt for aggregation
        prompt = f"summarize: {combined_text}"
        
        # Generate aggregated summary with optimal parameters
        final_summary = self.summarizer.summarize(
            prompt,
            max_length=400,
            min_length=150,  # Natural stopping point
            instructions=instructions
        )
        
        return final_summary
    
    def _recursive_aggregate(
        self,
        summaries: List[str],
        instructions: str = None,
        level: int = 0
    ) -> str:
        """
        Recursively aggregate summaries.
        
        Args:
            summaries: List of summaries
            instructions: Optional instructions
            level: Current recursion level
        
        Returns:
            Final summary
        """
        logger.info(f"Recursive aggregation level {level}: {len(summaries)} summaries")
        
        # Split into batches
        intermediate_summaries = []
        
        for i in range(0, len(summaries), self.max_chunks_per_level):
            batch = summaries[i:i + self.max_chunks_per_level]
            batch_summary = self._combine_summaries(batch, instructions=None)
            intermediate_summaries.append(batch_summary)
        
        # If we're down to one summary, we're done
        if len(intermediate_summaries) == 1:
            # Apply instructions at final level
            if instructions:
                return self.summarizer.refine_summary(
                    intermediate_summaries[0],
                    instructions
                )
            return intermediate_summaries[0]
        
        # Otherwise, recurse
        return self._recursive_aggregate(
            intermediate_summaries,
            instructions,
            level + 1
        )
    
    def aggregate_with_metadata(
        self,
        summaries: List[str],
        instructions: str = None
    ) -> dict:
        """
        Aggregate summaries and return with metadata.
        
        Args:
            summaries: List of summaries
            instructions: Optional instructions
        
        Returns:
            Dictionary with summary and metadata
        """
        final_summary = self.aggregate(summaries, instructions)
        
        return {
            "summary": final_summary,
            "num_chunks": len(summaries),
            "aggregation_method": "recursive" if len(summaries) > self.max_chunks_per_level else "direct",
            "chunk_summaries": summaries
        }


def aggregate_summaries(
    summaries: List[str],
    summarizer,
    instructions: str = None
) -> str:
    """
    Convenience function to aggregate summaries.
    
    Args:
        summaries: List of summaries to aggregate
        summarizer: Summarizer instance
        instructions: Optional user instructions
    
    Returns:
        Aggregated summary
    """
    aggregator = HierarchicalAggregator(summarizer)
    return aggregator.aggregate(summaries, instructions)


if __name__ == "__main__":
    # Test aggregator
    from model_loader import load_summarization_model
    from summarizer import Summarizer
    
    print("Loading model...")
    model, tokenizer, device = load_summarization_model()
    summarizer = Summarizer(model, tokenizer, device)
    
    print("Creating aggregator...")
    aggregator = HierarchicalAggregator(summarizer, max_chunks_per_level=3)
    
    # Test with multiple summaries
    test_summaries = [
        "AI is transforming healthcare with diagnostic tools.",
        "Machine learning improves financial fraud detection.",
        "Self-driving cars use deep learning for navigation.",
        "Natural language processing enables better chatbots.",
        "Computer vision helps in medical image analysis.",
        "Robotics benefits from reinforcement learning.",
        "AI assists in drug discovery and development."
    ]
    
    print(f"\nAggregating {len(test_summaries)} summaries...")
    final_summary = aggregator.aggregate(test_summaries)
    print(f"\nFinal summary: {final_summary}")
    
    print("\nWith metadata:")
    result = aggregator.aggregate_with_metadata(test_summaries)
    print(f"Summary: {result['summary']}")
    print(f"Method: {result['aggregation_method']}")
    print(f"Chunks: {result['num_chunks']}")
