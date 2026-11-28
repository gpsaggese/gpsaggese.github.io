import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm(temperature: float = 0.2):
    """Returns an LLM client pointed at NVIDIA's NIM endpoint."""
    return ChatOpenAI(
        model=os.getenv("NVIDIA_MODEL"),
        openai_api_key=os.getenv("NVIDIA_API_KEY"),
        openai_api_base=os.getenv("NVIDIA_BASE_URL"),
        temperature=temperature,
    )
