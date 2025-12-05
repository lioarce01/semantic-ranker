"""
Compression utilities for efficient reranking.

This module provides passage embedding compression (PECC) to reduce
token count while maintaining semantic information.
"""

from .pecc import PassageEmbeddingCompressor

__all__ = ['PassageEmbeddingCompressor']
