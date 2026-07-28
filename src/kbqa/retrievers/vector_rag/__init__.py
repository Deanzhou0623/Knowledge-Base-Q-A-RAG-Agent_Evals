"""Vector RAG retrieval backend: deterministic chunks embedded into FAISS.

Add backend-specific helpers (chunkers, embedding adapters, index builders) to
this package.
"""

from kbqa.retrievers.vector_rag.vector import VectorRetriever

__all__ = ["VectorRetriever"]
