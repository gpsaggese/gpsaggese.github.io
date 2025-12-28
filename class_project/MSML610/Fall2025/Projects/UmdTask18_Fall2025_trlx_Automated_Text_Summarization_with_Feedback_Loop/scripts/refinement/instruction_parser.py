"""
Instruction Parser

Parses user feedback and instructions for summary refinement.
Maps common patterns to prompt modifications.
"""

import re
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstructionParser:
    """Parse and interpret user instructions."""
    
    # Common instruction patterns
    PATTERNS = {
        "shorter": ["shorter", "brief", "concise", "condense"],
        "longer": ["longer", "detailed", "elaborate", "expand"],
        "bullet_points": ["bullet", "points", "list"],
        "headline": ["headline", "title", "one line"],
        "technical": ["technical", "detailed", "specific"],
        "simple": ["simple", "easy", "layman"],
        "formal": ["formal", "professional"],
        "casual": ["casual", "informal", "conversational"],
        "numbers": ["numbers", "statistics", "data", "figures"],
        "names": ["names", "people", "who"],
        "dates": ["dates", "when", "timeline"],
        "locations": ["locations", "where", "places"],
        "comparison": ["compare", "contrast", "difference"],
        "key_points": ["key points", "main points", "highlights"],
    }
    
    def parse(self, instruction: str) -> Dict[str, any]:
        """
        Parse user instruction.
        
        Args:
            instruction: User instruction text
        
        Returns:
            Dictionary with parsed instruction details
        """
        instruction_lower = instruction.lower()
        
        # Detect patterns
        detected_patterns = []
        for pattern_name, keywords in self.PATTERNS.items():
            if any(keyword in instruction_lower for keyword in keywords):
                detected_patterns.append(pattern_name)
        
        # Extract specific requests
        include_items = self._extract_include_items(instruction)
        exclude_items = self._extract_exclude_items(instruction)
        
        return {
            "original": instruction,
            "patterns": detected_patterns,
            "include": include_items,
            "exclude": exclude_items,
            "formatted_instruction": self._format_instruction(
                detected_patterns, include_items, exclude_items
            )
        }
    
    def _extract_include_items(self, instruction: str) -> List[str]:
        """Extract items to include."""
        include_patterns = [
            r"include\s+(.+?)(?:\.|$)",
            r"add\s+(.+?)(?:\.|$)",
            r"mention\s+(.+?)(?:\.|$)",
        ]
        
        items = []
        for pattern in include_patterns:
            matches = re.findall(pattern, instruction, re.IGNORECASE)
            items.extend(matches)
        
        return [item.strip() for item in items]
    
    def _extract_exclude_items(self, instruction: str) -> List[str]:
        """Extract items to exclude."""
        exclude_patterns = [
            r"exclude\s+(.+?)(?:\.|$)",
            r"remove\s+(.+?)(?:\.|$)",
            r"without\s+(.+?)(?:\.|$)",
            r"don't\s+(?:include|mention)\s+(.+?)(?:\.|$)",
        ]
        
        items = []
        for pattern in exclude_patterns:
            matches = re.findall(pattern, instruction, re.IGNORECASE)
            items.extend(matches)
        
        return [item.strip() for item in items]
    
    def _format_instruction(
        self,
        patterns: List[str],
        include: List[str],
        exclude: List[str]
    ) -> str:
        """Format instruction for model."""
        parts = []
        
        # Add style instructions
        if "shorter" in patterns:
            parts.append("Make it shorter and more concise")
        elif "longer" in patterns:
            parts.append("Provide more detail and elaboration")
        
        if "bullet_points" in patterns:
            parts.append("Format as bullet points")
        elif "headline" in patterns:
            parts.append("Create a single headline")
        
        if "technical" in patterns:
            parts.append("Use technical language and include specific details")
        elif "simple" in patterns:
            parts.append("Use simple language for general audience")
        
        if "formal" in patterns:
            parts.append("Use formal, professional tone")
        elif "casual" in patterns:
            parts.append("Use casual, conversational tone")
        
        # Add content instructions
        if "numbers" in patterns:
            parts.append("Include numbers and statistics")
        if "names" in patterns:
            parts.append("Include names of people")
        if "dates" in patterns:
            parts.append("Include dates and timeline")
        if "locations" in patterns:
            parts.append("Include locations and places")
        if "comparison" in patterns:
            parts.append("Compare and contrast key themes")
        if "key_points" in patterns:
            parts.append("Focus on key points and highlights")
        
        # Add include/exclude
        if include:
            parts.append(f"Include: {', '.join(include)}")
        if exclude:
            parts.append(f"Exclude: {', '.join(exclude)}")
        
        return ". ".join(parts) if parts else None


def parse_instruction(instruction: str) -> Dict[str, any]:
    """
    Convenience function to parse instruction.
    
    Args:
        instruction: User instruction
    
    Returns:
        Parsed instruction dictionary
    """
    parser = InstructionParser()
    return parser.parse(instruction)


if __name__ == "__main__":
    # Test instruction parser
    parser = InstructionParser()
    
    test_instructions = [
        "Make it shorter",
        "Format as bullet points and include numbers",
        "Make it more detailed and technical",
        "Create a headline",
        "Include financial data but exclude names",
        "Compare the key themes across all articles",
        "Make it simple and brief, suitable for general audience"
    ]
    
    for instruction in test_instructions:
        print(f"\nInstruction: {instruction}")
        result = parser.parse(instruction)
        print(f"Patterns: {result['patterns']}")
        print(f"Include: {result['include']}")
        print(f"Exclude: {result['exclude']}")
        print(f"Formatted: {result['formatted_instruction']}")
