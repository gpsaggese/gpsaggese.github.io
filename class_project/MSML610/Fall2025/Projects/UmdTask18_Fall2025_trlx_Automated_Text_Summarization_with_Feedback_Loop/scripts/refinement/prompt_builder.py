"""
Prompt Builder

Builds T5 prompts with instructions and formatting for different output styles.
"""

from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build prompts for T5 summarization."""
    
    @staticmethod
    def build_basic_prompt(text: str) -> str:
        """
        Build basic summarization prompt.
        
        Args:
            text: Input text
        
        Returns:
            Formatted prompt
        """
        return f"summarize: {text}"
    
    @staticmethod
    def build_instruction_prompt(text: str, instructions: str) -> str:
        """
        Build prompt with instructions.
        
        Args:
            text: Input text
            instructions: User instructions
        
        Returns:
            Formatted prompt with instructions
        """
        return f"summarize: {text}. Instructions: {instructions}"
    
    @staticmethod
    def build_refinement_prompt(summary: str, feedback: str) -> str:
        """
        Build prompt for refining existing summary.
        
        Args:
            summary: Existing summary
            feedback: User feedback
        
        Returns:
            Refinement prompt
        """
        return f"summarize: {summary}. Instructions: {feedback}"
    
    @staticmethod
    def build_multi_document_prompt(texts: list, instructions: Optional[str] = None) -> str:
        """
        Build prompt for multi-document summarization.
        
        Args:
            texts: List of texts
            instructions: Optional instructions
        
        Returns:
            Multi-document prompt
        """
        combined = " ".join(texts)
        
        if instructions:
            return f"summarize: {combined}. Instructions: {instructions}"
        else:
            return f"summarize: {combined}"
    
    @staticmethod
    def build_comparison_prompt(texts: list) -> str:
        """
        Build prompt for comparing multiple texts.
        
        Args:
            texts: List of texts to compare
        
        Returns:
            Comparison prompt
        """
        combined = " ".join(texts)
        return f"summarize: {combined}. Instructions: Compare and contrast the key themes across all texts"
    
    @staticmethod
    def build_style_prompt(text: str, style: str) -> str:
        """
        Build prompt with specific style.
        
        Args:
            text: Input text
            style: Desired style
        
        Returns:
            Styled prompt
        """
        style_instructions = {
            "bullet_points": "Format as bullet points",
            "headline": "Create a single headline",
            "brief": "Make it very brief, one sentence only",
            "detailed": "Provide a detailed summary with key points",
            "technical": "Use technical language and include specific details",
            "simple": "Use simple language suitable for general audience",
            "formal": "Use formal, professional tone",
            "casual": "Use casual, conversational tone"
        }
        
        instruction = style_instructions.get(style, None)
        
        if instruction:
            return f"summarize: {text}. Instructions: {instruction}"
        else:
            return f"summarize: {text}"


def build_prompt(
    text: str,
    instructions: Optional[str] = None,
    style: Optional[str] = None
) -> str:
    """
    Convenience function to build prompt.
    
    Args:
        text: Input text
        instructions: Optional user instructions
        style: Optional style
    
    Returns:
        Formatted prompt
    """
    builder = PromptBuilder()
    
    if style:
        return builder.build_style_prompt(text, style)
    elif instructions:
        return builder.build_instruction_prompt(text, instructions)
    else:
        return builder.build_basic_prompt(text)


if __name__ == "__main__":
    # Test prompt builder
    builder = PromptBuilder()
    
    test_text = "This is a sample article about artificial intelligence."
    
    print("Basic prompt:")
    print(builder.build_basic_prompt(test_text))
    
    print("\nWith instructions:")
    print(builder.build_instruction_prompt(test_text, "make it shorter"))
    
    print("\nWith style:")
    print(builder.build_style_prompt(test_text, "bullet_points"))
    
    print("\nRefinement:")
    print(builder.build_refinement_prompt("AI is advancing.", "add more details"))
    
    print("\nMulti-document:")
    texts = ["Article 1 about AI.", "Article 2 about ML."]
    print(builder.build_multi_document_prompt(texts, "compare themes"))
