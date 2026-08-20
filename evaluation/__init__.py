"""
evaluation/__init__.py
─────────────────────
Evaluation module for clinical severity RAG system.
"""
from evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    ClinicalAccuracyMetrics,
    PerformanceMetrics,
    ConsistencyMetrics,
    ComprehensiveRAGReport,
    evaluate_retrieval,
    evaluate_generation,
    evaluate_clinical_accuracy,
    get_rating,
)

__all__ = [
    "RetrievalMetrics",
    "GenerationMetrics",
    "ClinicalAccuracyMetrics",
    "PerformanceMetrics",
    "ConsistencyMetrics",
    "ComprehensiveRAGReport",
    "evaluate_retrieval",
    "evaluate_generation",
    "evaluate_clinical_accuracy",
    "get_rating",
]
