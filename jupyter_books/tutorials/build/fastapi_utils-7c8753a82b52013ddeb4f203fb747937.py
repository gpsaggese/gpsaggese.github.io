"""
Utility functions for the `FastAPI` + `uvicorn` tutorial notebooks.

- Notebooks should call these functions instead of writing raw logic inline
- Keeps `fastapi.API.ipynb` and `fastapi.example.ipynb` focused on
  explanation, with implementation details factored out here

Import as:

import tutorials.fastapi.fastapi_utils as tfasuti
"""

import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import helpers.hdbg as hdbg
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)


# #############################################################################
# Data models
# #############################################################################


class BookCreate(BaseModel):
    """
    Payload accepted when creating or replacing a book.
    """

    title: str = Field(..., description="Book title.")
    author: str = Field(..., description="Book author.")
    year: int = Field(..., ge=0, description="Publication year.")
    in_stock: bool = True


class Book(BaseModel):
    """
    A book as stored and returned by the API, including its ID.

    Declared as a standalone model (rather than `BookCreate` plus an `id`)
    so the required `id` field comes before the defaulted `in_stock` field,
    which keeps static type checkers happy about constructor call sites.
    """

    id: int = Field(..., description="Unique book identifier.")
    title: str = Field(..., description="Book title.")
    author: str = Field(..., description="Book author.")
    year: int = Field(..., ge=0, description="Publication year.")
    in_stock: bool = True


# #############################################################################
# Seed data
# #############################################################################


def build_seed_books() -> Dict[int, Book]:
    """
    Build a small in-memory catalog used to seed the demo app.

    :return: mapping from book ID to `Book`
    """
    seed = [
        Book(id=1, title="Fluent Python", author="Luciano Ramalho", year=2015),
        Book(id=2, title="Clean Code", author="Robert C. Martin", year=2008),
        Book(
            id=3,
            title="Designing Data-Intensive Applications",
            author="Martin Kleppmann",
            year=2017,
            in_stock=False,
        ),
    ]
    books = {book.id: book for book in seed}
    hdbg.dassert_lt(0, len(books), "Seed catalog must not be empty.")
    return books


# #############################################################################
# App factory
# #############################################################################


def create_book_app(seed: Optional[Dict[int, Book]] = None) -> FastAPI:
    """
    Build a small "Book Catalog" `FastAPI` app backed by an in-memory store.

    :param seed: initial catalog; if `None`, `build_seed_books()` is used
    :return: a ready-to-run `FastAPI` app
    """
    if seed is None:
        seed = build_seed_books()
    app = FastAPI(title="Book Catalog API")
    # Store mutable state on `app.state` instead of a module-level global, so
    # each call to `create_book_app()` gets its own isolated catalog.
    app.state.books = dict(seed)
    app.state.next_id = max(app.state.books, default=0) + 1

    @app.get("/health")
    def health() -> Dict[str, str]:
        """
        Report that the service is up.
        """
        return {"status": "ok"}

    @app.get("/books", response_model=List[Book])
    def list_books(
        in_stock: Optional[bool] = Query(
            None, description="Filter by stock availability."
        ),
        skip: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100),
    ) -> List[Book]:
        """
        List books, optionally filtered by stock, with pagination.
        """
        books = list(app.state.books.values())
        if in_stock is not None:
            books = [book for book in books if book.in_stock == in_stock]
        return books[skip : skip + limit]

    @app.get("/books/{book_id}", response_model=Book)
    def get_book(book_id: int) -> Book:
        """
        Fetch a single book by ID, or raise a 404.
        """
        return _get_book_or_404(app, book_id)

    @app.post("/books", response_model=Book, status_code=201)
    def create_book(payload: BookCreate) -> Book:
        """
        Add a new book to the catalog.
        """
        book_id = app.state.next_id
        book = Book(id=book_id, **payload.model_dump())
        app.state.books[book_id] = book
        app.state.next_id += 1
        return book

    @app.patch("/books/{book_id}", response_model=Book)
    def update_book(book_id: int, payload: BookCreate) -> Book:
        """
        Replace an existing book's fields.
        """
        _get_book_or_404(app, book_id)
        book = Book(id=book_id, **payload.model_dump())
        app.state.books[book_id] = book
        return book

    @app.delete("/books/{book_id}", status_code=204)
    def delete_book(book_id: int) -> None:
        """
        Remove a book from the catalog.
        """
        _get_book_or_404(app, book_id)
        del app.state.books[book_id]

    return app


# Module-level app so `uvicorn fastapi_utils:app --reload` works directly
# from the command line, outside of the notebooks.
app = create_book_app()


def _get_book_or_404(app: FastAPI, book_id: int) -> Book:
    """
    Look up a book or raise the `FastAPI` 404 error.

    :param app: app whose `state.books` store is searched
    :param book_id: ID to look up
    :return: the matching `Book`
    """
    if book_id not in app.state.books:
        raise HTTPException(status_code=404, detail=f"Book {book_id} not found")
    return app.state.books[book_id]


# #############################################################################
# Test / server helpers
# #############################################################################


def run_server_in_background(
    app: FastAPI, *, host: str = "127.0.0.1", port: int = 8000
) -> Tuple[uvicorn.Server, threading.Thread]:
    """
    Start `app` with `uvicorn` on a background thread.

    Useful in a notebook, where nothing else can run while `uvicorn.run()`
    blocks the foreground.

    :param app: `FastAPI` app to serve
    :param host: interface to bind to
    :param port: port to bind to
    :return: the `uvicorn.Server` instance and the thread running it
    """
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _LOG.info(
        "Started uvicorn on 'http://%s:%s' in a background thread.", host, port
    )
    return server, thread


def stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    """
    Signal a background `uvicorn.Server` to stop and wait for its thread.

    :param server: server returned by `run_server_in_background()`
    :param thread: thread returned by `run_server_in_background()`
    """
    server.should_exit = True
    thread.join(timeout=5.0)
    _LOG.info("Stopped background uvicorn server.")


def wait_for_server(url: str, *, timeout: float = 5.0) -> None:
    """
    Poll `url` until it responds or `timeout` seconds elapse.

    :param url: URL to poll, e.g. the app's `/health` endpoint
    :param timeout: max seconds to wait before giving up
    """
    deadline = time.monotonic() + timeout
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError as e:
            last_error = e
        time.sleep(0.1)
    hdbg.dfatal(f"Server at '{url}' did not become ready: {last_error}")
