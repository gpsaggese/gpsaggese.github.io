import hashlib
import os
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


def list_markdown_files(root_dir: str | Path) -> list[Path]:
    """
    Recursively list markdown files under `root_dir`.
    """
    root = Path(root_dir).resolve()
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def file_checksum(path: str | Path) -> str:
    """
    Compute a stable checksum for a file.
    """
    resolved = Path(path).resolve()
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def snapshot_checksums(paths: Iterable[str | Path]) -> dict[str, str]:
    """
    Build a `{absolute_path: checksum}` snapshot for change detection.
    """
    snapshot: dict[str, str] = {}
    for path in paths:
        resolved = Path(path).resolve()
        snapshot[str(resolved)] = file_checksum(resolved)
    return snapshot


def diff_checksum_snapshots(
    previous: dict[str, str], current: dict[str, str]
) -> dict[str, list[str]]:
    """
    Diff two checksum snapshots and return changed file groups.
    """
    previous_paths = set(previous.keys())
    current_paths = set(current.keys())
    new_files = sorted(current_paths - previous_paths)
    deleted_files = sorted(previous_paths - current_paths)
    modified_files = sorted(
        path
        for path in (previous_paths & current_paths)
        if previous[path] != current[path]
    )
    return {
        "new": new_files,
        "modified": modified_files,
        "deleted": deleted_files,
    }


def load_markdown_documents(paths: Iterable[str | Path]) -> list[Document]:
    """
    Load markdown files into LangChain `Document` objects.
    """
    documents: list[Document] = []
    for path in paths:
        resolved = Path(path).resolve()
        text = resolved.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(resolved),
                    "checksum": file_checksum(resolved),
                },
            )
        )
    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Document]:
    """
    Split documents into overlapping character chunks.
    """
    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))

    chunked: list[Document] = []
    for document in documents:
        text = document.page_content
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            metadata = dict(document.metadata)
            metadata["chunk_index"] = chunk_index
            chunked.append(Document(page_content=chunk_text, metadata=metadata))
            if end >= len(text):
                break
            start = end - chunk_overlap
            chunk_index += 1
    return chunked


def make_embeddings() -> Embeddings:
    """
    Build embeddings from env config.

    Behavior:
    - `EMBEDDING_PROVIDER=openai` (or `auto` with `OPENAI_API_KEY`) -> OpenAI embeddings
    - otherwise -> deterministic fake embeddings (cheap, reproducible, lower quality)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
    if provider in {"openai", "auto"} and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    size = int(os.getenv("FAKE_EMBED_DIM", "256"))
    return DeterministicFakeEmbedding(size=size)


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
) -> InMemoryVectorStore:
    """
    Build an in-memory vector store from chunked documents.
    """
    store = InMemoryVectorStore(embedding=embeddings)
    if documents:
        store.add_documents(documents)
    return store


def add_documents_to_store(
    store: InMemoryVectorStore, documents: list[Document]
) -> None:
    """
    Append chunked documents to an existing vector store.
    """
    if documents:
        store.add_documents(documents)


def format_docs(documents: list[Document]) -> str:
    """
    Render retrieved documents as a compact context string.
    """
    blocks: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "(unknown)")
        chunk = document.metadata.get("chunk_index", "?")
        blocks.append(f"[source={source} chunk={chunk}]\n{document.page_content}")
    return "\n\n---\n\n".join(blocks)
