import torch
from transformers import pipeline


summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
)


def summarize_text(text: str) -> str:
    """
    Summarize full transcript.
    BART works best when input is not too long; we chunk if needed.
    """
    text = (text or "").strip()
    if len(text.split()) < 20:
        return "Text too short to summarize."

    # BART max input is ~1024 tokens.
    words = text.split()
    chunk_size_words = 450 
    chunks = [" ".join(words[i:i + chunk_size_words]) for i in range(0, len(words), chunk_size_words)]

    partial_summaries = []
    for c in chunks:
        out = summarizer(c, max_length=150, min_length=40, do_sample=False)
        partial_summaries.append(out[0]["summary_text"])

    # If multiple chunks, summarize the summaries again for a final compact summary
    if len(partial_summaries) > 1:
        combined = " ".join(partial_summaries)
        out = summarizer(combined, max_length=160, min_length=50, do_sample=False)
        return out[0]["summary_text"]

    return partial_summaries[0]
