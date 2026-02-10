"""
Server-side handlers for the Context Chat JupyterLab extension.

This module provides REST endpoints for the chat functionality:
- POST /context-chat/generate: Generates AI responses using LangChain
"""

import json
import os
import logging
from typing import Dict, Any
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in parent directories
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded environment from {env_path}")
except ImportError:
    logging.info("python-dotenv not installed, using system environment variables")

from tornado.web import RequestHandler

# Disable CSRF for API endpoints (Jupyter handles auth via tokens)
from jupyter_server.base.handlers import APIHandler

# Optional: Import LangChain for production use
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Using stub responses.")


def generate_reply_stub(message: str, context: str) -> str:
    """
    Stub response generator for development/testing.
    
    Replace this with actual LangChain call in production.
    """
    response_parts = [
        "I received your message and notebook context.",
        "",
        f"**Your message:** {message}",
        "",
        "**Context preview (first 800 chars):**",
        f"```\n{context[:800]}\n```",
        "",
        "Note: This is a stub response. Configure your OpenAI API key and",
        "install langchain packages to enable real AI responses."
    ]
    
    return "\n".join(response_parts)


def generate_reply_langchain(message: str, context: str) -> str:
    """
    Generate AI response using LangChain and OpenAI.
    
    Args:
        message: User's question/message
        context: Notebook context (cell content, variables, etc.)
    
    Returns:
        AI-generated response string
    """
    try:
        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        )
        
        # Build messages with context
        messages = []
        
        # System prompt
        system_prompt = """You are an AI assistant embedded in JupyterLab.
You help users understand and work with their Jupyter notebooks.
You have access to the user's notebook context (cells, code, markdown).
Be helpful, concise, and provide actionable advice.
When discussing code, use markdown code blocks with appropriate language tags."""
        
        messages.append(SystemMessage(content=system_prompt))
        
        # Add context if provided
        if context and context.strip():
            context_msg = f"**Current notebook context:**\n\n{context}"
            messages.append(SystemMessage(content=context_msg))
        
        # Add user message
        messages.append(HumanMessage(content=message))
        
        # Get response
        response = llm.invoke(messages)
        
        return response.content
        
    except Exception as e:
        logging.error(f"LangChain error: {e}")
        return f"Error generating response: {str(e)}\n\nPlease check your OpenAI API key and LangChain installation."


def generate_reply(message: str, context: str) -> str:
    """
    Main entry point for generating replies.
    
    Uses LangChain if available, otherwise returns stub response.
    """
    if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        return generate_reply_langchain(message, context)
    else:
        return generate_reply_stub(message, context)


class GenerateHandler(APIHandler):
    """
    Tornado handler for /context-chat/generate endpoint.
    
    Uses APIHandler instead of RequestHandler to integrate with Jupyter's
    authentication and CORS handling.
    
    Accepts POST requests with JSON body:
    {
        "message": "user question",
        "context": "notebook context (optional)"
    }
    
    Returns JSON:
    {
        "reply": "AI response"
    }
    """
    
    def post(self):
        """Handle POST request to generate AI response."""
        try:
            # Parse request body using APIHandler's method
            data = self.get_json_body()
            message = (data.get("message") or "").strip()
            context = (data.get("context") or "").strip()
            
            # Validate input
            if not message:
                self.set_status(400)
                self.finish(json.dumps({"error": "message is required"}))
                return
            
            # Generate response
            logging.info(f"Generating response for message: {message[:100]}...")
            reply = generate_reply(message, context)
            
            # Return response
            self.finish(json.dumps({"reply": reply}))
            
        except Exception as e:
            logging.error(f"Error in GenerateHandler: {e}")
            self.set_status(500)
            self.finish(json.dumps({"error": f"Server error: {str(e)}"}))
