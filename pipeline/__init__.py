"""
LLM-Powered Information Extraction Pipeline

This package provides tools for extracting insights and policy conclusions 
from long research documents using chained LLM calls.
"""

from .document_processor import DocumentProcessor
from .llm_chains import LLMChainManager, ChainOutputParser
from .evaluation import EvaluationMetrics
from .utils import ResultsExporter, CacheManager

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "DocumentProcessor",
    "LLMChainManager", 
    "ChainOutputParser",
    "EvaluationMetrics",
    "ResultsExporter",
    "CacheManager"
]