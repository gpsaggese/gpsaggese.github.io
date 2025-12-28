"""
Web Backend for RLHF News Summarization System

FastAPI backend that integrates with the modular summarization pipeline.
Supports text, URL, and file-based summarization with refinement.
"""

# CRITICAL: Set environment variables BEFORE any imports
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path
import tempfile
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    summarize_text,
    summarize_url,
    summarize_urls,
    summarize_file,
    summarize_files,
    refine_summary,
    get_model_info
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RLHF News Summarization API",
    description="API for news summarization using DPO-trained T5-large",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class TextSummarizeRequest(BaseModel):
    text: str
    instructions: Optional[str] = None

class URLSummarizeRequest(BaseModel):
    url: str
    instructions: Optional[str] = None

class MultiURLSummarizeRequest(BaseModel):
    urls: List[str]
    instructions: Optional[str] = None
    combine: bool = True

class RefinementRequest(BaseModel):
    summary: str
    feedback: str
    original_text: Optional[str] = None


# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RLHF News Summarization API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/model/info": "Model information",
            "/summarize/text": "Summarize text",
            "/summarize/url": "Summarize from URL",
            "/summarize/urls": "Summarize multiple URLs",
            "/summarize/file": "Summarize uploaded file",
            "/refine": "Refine existing summary"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        info = get_model_info()
        return {
            "status": "healthy",
            "model_loaded": info["loaded"],
            "device": info["device"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/model/info")
async def model_info():
    """Get model information."""
    try:
        info = get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/text")
async def api_summarize_text(request: TextSummarizeRequest):
    """Summarize text."""
    try:
        logger.info(f"Summarizing text ({len(request.text)} chars)")
        result = summarize_text(request.text, request.instructions)
        
        return {
            "success": True,
            "summary": result["summary"],
            "num_chunks": result["num_chunks"],
            "input_length": result["input_length"],
            "summary_length": result["summary_length"],
            "chunk_summaries": result.get("chunk_summaries", [])
        }
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/url")
async def api_summarize_url(request: URLSummarizeRequest):
    """Summarize from URL."""
    try:
        logger.info(f"Summarizing URL: {request.url}")
        result = summarize_url(request.url, request.instructions)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to extract from URL")
            )
        
        return {
            "success": True,
            "summary": result["summary"],
            "url": result["url"],
            "title": result.get("title"),
            "author": result.get("author"),
            "date": result.get("date"),
            "num_chunks": result["num_chunks"],
            "input_length": result["input_length"],
            "summary_length": result["summary_length"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/urls")
async def api_summarize_urls(request: MultiURLSummarizeRequest):
    """Summarize multiple URLs."""
    try:
        logger.info(f"Summarizing {len(request.urls)} URLs")
        result = summarize_urls(request.urls, request.instructions, request.combine)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to extract from URLs")
            )
        
        return {
            "success": True,
            "summary": result.get("summary"),
            "summaries": result.get("summaries"),
            "num_urls": result["num_urls"],
            "successful_extractions": result["successful_extractions"],
            "url_results": result.get("url_results", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing URLs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/file")
async def api_summarize_file(
    file: UploadFile = File(...),
    instructions: Optional[str] = Form(None)
):
    """Summarize uploaded file (PDF, DOCX, TXT)."""
    try:
        logger.info(f"Summarizing file: {file.filename}")
        
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Summarize
        result = summarize_file(tmp_path, instructions)
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to extract from file")
            )
        
        return {
            "success": True,
            "summary": result["summary"],
            "filename": file.filename,
            "file_type": result["file_type"],
            "num_chunks": result["num_chunks"],
            "input_length": result["input_length"],
            "summary_length": result["summary_length"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/files")
async def api_summarize_files(
    files: List[UploadFile] = File(...),
    instructions: Optional[str] = Form(None),
    combine: bool = Form(True)
):
    """Summarize multiple uploaded files (PDF, DOCX, TXT)."""
    try:
        logger.info(f"Summarizing {len(files)} files")
        
        # Save all uploaded files to temp locations
        tmp_paths = []
        filenames = []
        
        for file in files:
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_paths.append(tmp_file.name)
                filenames.append(file.filename)
        
        # Summarize
        result = summarize_files(tmp_paths, instructions, combine)
        
        # Clean up temp files
        for tmp_path in tmp_paths:
            Path(tmp_path).unlink()
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to extract from files")
            )
        
        return {
            "success": True,
            "summary": result.get("summary"),
            "summaries": result.get("summaries"),
            "filenames": filenames,
            "num_files": result["num_files"],
            "successful_extractions": result["successful_extractions"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/refine")
async def api_refine_summary(request: RefinementRequest):
    """Refine existing summary."""
    try:
        logger.info(f"Refining summary with feedback: {request.feedback}")
        result = refine_summary(
            request.summary,
            request.feedback,
            request.original_text
        )
        
        return {
            "success": True,
            "refined_summary": result["refined_summary"],
            "original_summary": result["original_summary"],
            "feedback": result["feedback"]
        }
    except Exception as e:
        logger.error(f"Error refining summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("Starting RLHF News Summarization API...")
    print("Loading model...")
    
    # Reset pipeline to force reload with latest parameters
    from scripts.pipeline.summarization_pipeline import reset_pipeline
    reset_pipeline()
    
    # Pre-load model
    try:
        info = get_model_info()
        print(f"Model loaded: {info['model_name']}")
        print(f"Device: {info['device']}")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
    
    print("\nAPI will be available at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
