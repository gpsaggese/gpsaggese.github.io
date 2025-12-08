"""
Configuration settings for the NBA data analysis project.
"""
import os
from crewai import LLM

# NBA Data Configuration
NBA_DATA_PATH = "nba24-25.csv"

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2"  

# Set the base URL for Ollama
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL


def get_llm() -> LLM:
    """
    Create and return a CrewAI LLM instance configured for Ollama.
    
    Returns:
        LLM: Configured CrewAI LLM instance for Ollama
    """
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL
    )