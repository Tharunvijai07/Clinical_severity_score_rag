"""
evaluation/metrics.py
─────────────────────
Automated evaluation metrics for the clinical severity RAG system.

Metrics include:
  • Retrieval: Precision, Recall, NDCG, MRR
  • LLM Output: JSON validity, schema compliance
  • Clinical: Accuracy, F1-score against ground truth
  • Performance: Latency, throughput
  • Consistency: Output stability
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rag.retriever import RetrievedChunk
from llm.severity_analyzer import SeverityResult
from utils.logger import get_logger

logger = get_logger(__name__)


def normalize_source_name(source: str) -> str:
    """Normalize stems and filenames so evaluator labels match metadata."""
    return Path(source).stem.strip().lower()


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    precision_at_k: dict[int, float] = field(default_factory=dict)  # P@k
    recall_at_k: dict[int, float] = field(default_factory=dict)     # R@k
    ndcg_at_k: dict[int, float] = field(default_factory=dict)       # NDCG@k
    mrr: float = 0.0                                                 # Mean Reciprocal Rank
    avg_similarity: float = 0.0                                      # Avg cosine similarity
    
    def __str__(self) -> str:
        lines = ["RetrievalMetrics:"]
        for k, p in sorted(self.precision_at_k.items()):
            lines.append(f"  P@{k}: {p:.3f}")
        for k, r in sorted(self.recall_at_k.items()):
            lines.append(f"  R@{k}: {r:.3f}")
        for k, ndcg in sorted(self.ndcg_at_k.items()):
            lines.append(f"  NDCG@{k}: {ndcg:.3f}")
        lines.append(f"  MRR: {self.mrr:.3f}")
        lines.append(f"  Avg Similarity: {self.avg_similarity:.3f}")
        return "\n".join(lines)


def compute_precision_at_k(
    retrieved: list[RetrievedChunk],
    relevant_sources: set[str],
    k: int = 5,
) -> float:
    """
    Precision @ K: proportion of top-K results that are relevant.
    
    P@K = (# relevant in top K) / K
    
    Args:
        retrieved: List of RetrievedChunk objects.
        relevant_sources: Set of source filenames that are considered relevant.
        k: Cutoff rank.
    
    Returns:
        Precision score in [0, 1].
    """
    if k == 0:
        return 0.0
    
    top_k = retrieved[:k]
    relevant = {normalize_source_name(source) for source in relevant_sources}
    relevant_count = sum(
        1 for chunk in top_k if normalize_source_name(chunk.source) in relevant
    )
    return relevant_count / k


def compute_recall_at_k(
    retrieved: list[RetrievedChunk],
    relevant_sources: set[str],
    k: int = 5,
) -> float:
    """
    Recall @ K: proportion of all relevant items that appear in top-K.
    
    R@K = (# relevant in top K) / (total # relevant)
    
    Args:
        retrieved: List of RetrievedChunk objects.
        relevant_sources: Set of source filenames that are considered relevant.
        k: Cutoff rank.
    
    Returns:
        Recall score in [0, 1].
    """
    if not relevant_sources:
        return 1.0  # No relevant items means perfect recall
    
    top_k = retrieved[:k]
    relevant = {normalize_source_name(source) for source in relevant_sources}
    relevant_count = sum(
        1 for chunk in top_k if normalize_source_name(chunk.source) in relevant
    )
    return relevant_count / len(relevant_sources)


def compute_ndcg_at_k(
    retrieved: list[RetrievedChunk],
    relevant_sources: set[str],
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.
    
    Rewards ranking relevant items higher in the list.
    Uses similarity score as relevance signal.
    
    NDCG@K = DCG@K / IDCG@K
    DCG@K = sum(rel_i / log2(i+1)) for i in 1..K
    
    Args:
        retrieved: List of RetrievedChunk objects (ranked by relevance).
        relevant_sources: Set of source filenames that are considered relevant.
        k: Cutoff rank.
    
    Returns:
        NDCG score in [0, 1].
    """
    if not relevant_sources:
        return 1.0
    
    top_k = retrieved[:k]
    relevant = {normalize_source_name(source) for source in relevant_sources}
    
    # DCG: sum relevance weighted by position
    dcg = 0.0
    for i, chunk in enumerate(top_k, 1):
        relevance = 1.0 if normalize_source_name(chunk.source) in relevant else 0.0
        dcg += relevance / np.log2(i + 1)
    
    # IDCG: ideal DCG (all relevant items ranked first)
    ideal_relevances = [1.0] * min(len(relevant_sources), k)
    idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_relevances, 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_mrr(
    retrieved: list[RetrievedChunk],
    relevant_sources: set[str],
) -> float:
    """
    Mean Reciprocal Rank: position of first relevant item.
    
    MRR = 1 / rank_of_first_relevant_item
    
    Args:
        retrieved: List of RetrievedChunk objects (ranked by relevance).
        relevant_sources: Set of source filenames that are considered relevant.
    
    Returns:
        MRR score in [0, 1].
    """
    relevant = {normalize_source_name(source) for source in relevant_sources}
    for i, chunk in enumerate(retrieved, 1):
        if normalize_source_name(chunk.source) in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(
    retrieved: list[RetrievedChunk],
    relevant_sources: set[str],
    k_values: list[int] = None,
) -> RetrievalMetrics:
    """
    Compute all retrieval metrics for a single query.
    
    Args:
        retrieved: List of RetrievedChunk objects.
        relevant_sources: Set of source filenames that are considered relevant.
        k_values: List of K values for P@K, R@K, NDCG@K (default: [1, 3, 5, 10]).
    
    Returns:
        RetrievalMetrics dataclass.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    metrics = RetrievalMetrics()
    
    for k in k_values:
        metrics.precision_at_k[k] = compute_precision_at_k(retrieved, relevant_sources, k)
        metrics.recall_at_k[k] = compute_recall_at_k(retrieved, relevant_sources, k)
        metrics.ndcg_at_k[k] = compute_ndcg_at_k(retrieved, relevant_sources, k)
    
    metrics.mrr = compute_mrr(retrieved, relevant_sources)
    metrics.avg_similarity = np.mean([c.similarity for c in retrieved]) if retrieved else 0.0
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# LLM OUTPUT METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LLMOutputMetrics:
    """Metrics for evaluating LLM output quality."""
    
    json_validity_rate: float = 0.0     # % of outputs with valid JSON
    schema_compliance_rate: float = 0.0 # % of outputs matching schema
    score_range_compliance: float = 0.0 # % with score in [0, 10]
    level_compliance: float = 0.0       # % with valid severity level
    avg_parse_errors: float = 0.0       # Average # of parse errors per output
    
    def __str__(self) -> str:
        return f"""LLMOutputMetrics:
  JSON Validity: {self.json_validity_rate:.1%}
  Schema Compliance: {self.schema_compliance_rate:.1%}
  Score Range [0-10]: {self.score_range_compliance:.1%}
  Valid Severity Level: {self.level_compliance:.1%}
  Avg Parse Errors: {self.avg_parse_errors:.2f}"""


def evaluate_llm_output_batch(
    results: list[SeverityResult],
) -> LLMOutputMetrics:
    """
    Evaluate LLM output quality across a batch of results.
    
    Args:
        results: List of SeverityResult objects.
    
    Returns:
        LLMOutputMetrics dataclass.
    """
    if not results:
        return LLMOutputMetrics()
    
    n = len(results)
    valid_count = sum(1 for r in results if r.is_valid)
    
    # Schema compliance checks
    valid_levels = {"Low", "Moderate", "High", "Critical"}
    schema_compliant = 0
    score_compliant = 0
    level_compliant = 0
    parse_errors = []
    
    for result in results:
        if result.is_valid:
            schema_compliant += 1
        
        if 0 <= result.severity_score <= 10:
            score_compliant += 1
        
        if result.severity_level in valid_levels:
            level_compliant += 1
        
        # Count parse errors
        error_count = 1 if result.parse_error else 0
        parse_errors.append(error_count)
    
    metrics = LLMOutputMetrics(
        json_validity_rate=valid_count / n,
        schema_compliance_rate=schema_compliant / n,
        score_range_compliance=score_compliant / n,
        level_compliance=level_compliant / n,
        avg_parse_errors=np.mean(parse_errors),
    )
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CLINICAL ACCURACY METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ClinicalAccuracyMetrics:
    """Metrics for evaluating clinical accuracy against ground truth."""
    
    accuracy: float = 0.0              # % of correct severity levels
    precision: dict[str, float] = field(default_factory=dict)  # Per-class precision
    recall: dict[str, float] = field(default_factory=dict)     # Per-class recall
    f1: dict[str, float] = field(default_factory=dict)         # Per-class F1
    score_mae: float = 0.0              # Mean absolute error of severity score
    score_rmse: float = 0.0             # Root mean square error of severity score
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    
    def __str__(self) -> str:
        lines = [f"ClinicalAccuracyMetrics:\n  Accuracy: {self.accuracy:.1%}"]
        lines.append(f"  Score MAE: {self.score_mae:.2f}")
        lines.append(f"  Score RMSE: {self.score_rmse:.2f}")
        for level in ["Low", "Moderate", "High", "Critical"]:
            if level in self.precision:
                lines.append(f"  {level}: P={self.precision[level]:.3f} R={self.recall[level]:.3f} F1={self.f1[level]:.3f}")
        return "\n".join(lines)


def evaluate_clinical_accuracy(
    predictions: list[str],  # predicted severity levels
    ground_truth: list[str], # ground truth severity levels
    pred_scores: list[int] = None,  # predicted severity scores
    gt_scores: list[int] = None,    # ground truth severity scores
) -> ClinicalAccuracyMetrics:
    """
    Evaluate clinical accuracy against ground truth labels.
    
    Args:
        predictions: List of predicted severity levels.
        ground_truth: List of ground truth severity levels.
        pred_scores: Optional list of predicted severity scores.
        gt_scores: Optional list of ground truth severity scores.
    
    Returns:
        ClinicalAccuracyMetrics dataclass.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    n = len(predictions)
    metrics = ClinicalAccuracyMetrics()
    
    # Accuracy: % correct
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    metrics.accuracy = correct / n
    
    # Score metrics
    if pred_scores and gt_scores:
        if len(pred_scores) == n and len(gt_scores) == n:
            score_diffs = [abs(p - g) for p, g in zip(pred_scores, gt_scores)]
            metrics.score_mae = np.mean(score_diffs)
            metrics.score_rmse = np.sqrt(np.mean([d**2 for d in score_diffs]))
    
    # Per-class precision, recall, F1
    valid_levels = {"Low", "Moderate", "High", "Critical"}
    for level in valid_levels:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == level and g == level)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == level and g != level)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != level and g == level)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.precision[level] = precision
        metrics.recall[level] = recall
        metrics.f1[level] = f1
    
    # Confusion matrix
    for level in valid_levels:
        metrics.confusion_matrix[level] = {}
        for pred_level in valid_levels:
            count = sum(1 for p, g in zip(predictions, ground_truth) if g == level and p == pred_level)
            metrics.confusion_matrix[level][pred_level] = count
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PerformanceMetrics:
    """Metrics for latency and throughput."""
    
    retrieval_latency_ms: float = 0.0   # Retrieval time
    llm_latency_ms: float = 0.0         # LLM inference time
    total_latency_ms: float = 0.0       # Total end-to-end time
    queries_per_second: float = 0.0     # Throughput
    
    def __str__(self) -> str:
        return f"""PerformanceMetrics:
  Retrieval Latency: {self.retrieval_latency_ms:.2f} ms
  LLM Latency: {self.llm_latency_ms:.2f} ms
  Total Latency: {self.total_latency_ms:.2f} ms
  Throughput: {self.queries_per_second:.2f} q/s"""


# ═══════════════════════════════════════════════════════════════════════════════
# CONSISTENCY METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConsistencyMetrics:
    """Metrics for output consistency across multiple runs."""
    
    level_consistency: float = 0.0  # % of runs with same severity level
    score_std_dev: float = 0.0      # Std dev of severity scores
    
    def __str__(self) -> str:
        return f"""ConsistencyMetrics:
  Level Consistency: {self.level_consistency:.1%}
  Score Std Dev: {self.score_std_dev:.2f}"""


def evaluate_consistency(
    repeated_results: list[list[SeverityResult]],
) -> ConsistencyMetrics:
    """
    Evaluate consistency of outputs across multiple runs of same query.
    
    Args:
        repeated_results: List of result lists, one per run.
                         Each inner list contains results for same queries.
    
    Returns:
        ConsistencyMetrics dataclass.
    """
    if not repeated_results or not repeated_results[0]:
        return ConsistencyMetrics()
    
    n_queries = len(repeated_results[0])
    n_runs = len(repeated_results)
    
    # Level consistency: for each query, what % of runs agree?
    level_consistencies = []
    score_stds = []
    
    for q_idx in range(n_queries):
        levels = [repeated_results[r][q_idx].severity_level for r in range(n_runs)]
        scores = [repeated_results[r][q_idx].severity_score for r in range(n_runs)]
        
        # Most common level
        most_common = max(set(levels), key=levels.count)
        consistency = levels.count(most_common) / n_runs
        level_consistencies.append(consistency)
        
        # Score std dev
        score_stds.append(np.std(scores))
    
    metrics = ConsistencyMetrics(
        level_consistency=np.mean(level_consistencies),
        score_std_dev=np.mean(score_stds),
    )
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE EVALUATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvaluationReport:
    """Full evaluation report combining all metrics."""
    
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    llm_output: LLMOutputMetrics = field(default_factory=LLMOutputMetrics)
    clinical_accuracy: ClinicalAccuracyMetrics = field(default_factory=ClinicalAccuracyMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    consistency: ConsistencyMetrics = field(default_factory=ConsistencyMetrics)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def __str__(self) -> str:
        return f"""
╔════════════════════════════════════════════════════════════════════╗
║               CLINICAL SEVERITY RAG - EVALUATION REPORT            ║
║                         {self.timestamp}                        ║
╚════════════════════════════════════════════════════════════════════╝

{self.retrieval}

{self.llm_output}

{self.clinical_accuracy}

{self.performance}

{self.consistency}
"""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for logging/JSON export."""
        return {
            "timestamp": self.timestamp,
            "retrieval": {
                "precision_at_k": self.retrieval.precision_at_k,
                "recall_at_k": self.retrieval.recall_at_k,
                "ndcg_at_k": self.retrieval.ndcg_at_k,
                "mrr": self.retrieval.mrr,
                "avg_similarity": self.retrieval.avg_similarity,
            },
            "llm_output": {
                "json_validity_rate": self.llm_output.json_validity_rate,
                "schema_compliance_rate": self.llm_output.schema_compliance_rate,
                "score_range_compliance": self.llm_output.score_range_compliance,
                "level_compliance": self.llm_output.level_compliance,
                "avg_parse_errors": self.llm_output.avg_parse_errors,
            },
            "clinical_accuracy": {
                "accuracy": self.clinical_accuracy.accuracy,
                "score_mae": self.clinical_accuracy.score_mae,
                "score_rmse": self.clinical_accuracy.score_rmse,
                "precision": self.clinical_accuracy.precision,
                "recall": self.clinical_accuracy.recall,
                "f1": self.clinical_accuracy.f1,
            },
            "performance": {
                "retrieval_latency_ms": self.performance.retrieval_latency_ms,
                "llm_latency_ms": self.performance.llm_latency_ms,
                "total_latency_ms": self.performance.total_latency_ms,
                "queries_per_second": self.performance.queries_per_second,
            },
            "consistency": {
                "level_consistency": self.consistency.level_consistency,
                "score_std_dev": self.consistency.score_std_dev,
            },
        }
