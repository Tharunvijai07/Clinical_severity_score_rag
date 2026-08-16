"""
evaluation/__init__.py
─────────────────────
Evaluation module for clinical severity RAG system.

Export key components for easy access.
"""
from evaluation.metrics import (
    RetrievalMetrics,
    LLMOutputMetrics,
    ClinicalAccuracyMetrics,
    PerformanceMetrics,
    ConsistencyMetrics,
    EvaluationReport,
    evaluate_retrieval,
    evaluate_llm_output_batch,
    evaluate_clinical_accuracy,
    evaluate_consistency,
)

__all__ = [
    "RetrievalMetrics",
    "LLMOutputMetrics",
    "ClinicalAccuracyMetrics",
    "PerformanceMetrics",
    "ConsistencyMetrics",
    "EvaluationReport",
    "evaluate_retrieval",
    "evaluate_llm_output_batch",
    "evaluate_clinical_accuracy",
    "evaluate_consistency",
]
