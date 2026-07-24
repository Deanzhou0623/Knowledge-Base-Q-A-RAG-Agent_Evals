"""Markdown KB retrieval backend: heading sections retrieved with BM25.

Add backend-specific helpers (tokenizers, scorers, analysers) to this package.
"""

from kbqa.retrievers.markdown_kb.bm25 import BM25Retriever

__all__ = ["BM25Retriever"]
