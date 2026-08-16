"""
evaluation/test_suite.py
─────────────────────────
Automated test suite using evaluation metrics.

Usage:
    python -m evaluation.test_suite
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from rag.rag_pipeline import RAGPipeline
from llm.severity_analyzer import SeverityAnalyzer
from evaluation.metrics import (
    evaluate_retrieval,
    evaluate_llm_output_batch,
    evaluate_clinical_accuracy,
    PerformanceMetrics,
    ConsistencyMetrics,
    RetrievalMetrics,
    EvaluationReport,
    normalize_source_name,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _average_retrieval_metrics(results: dict) -> RetrievalMetrics:
    """Average per-case retrieval metrics into a report-level summary."""
    metrics_list = [item["metrics"] for item in results.values()]
    if not metrics_list:
        return RetrievalMetrics()

    k_values = sorted({k for metrics in metrics_list for k in metrics.precision_at_k})
    return RetrievalMetrics(
        precision_at_k={
            k: sum(metrics.precision_at_k.get(k, 0.0) for metrics in metrics_list) / len(metrics_list)
            for k in k_values
        },
        recall_at_k={
            k: sum(metrics.recall_at_k.get(k, 0.0) for metrics in metrics_list) / len(metrics_list)
            for k in k_values
        },
        ndcg_at_k={
            k: sum(metrics.ndcg_at_k.get(k, 0.0) for metrics in metrics_list) / len(metrics_list)
            for k in k_values
        },
        mrr=sum(metrics.mrr for metrics in metrics_list) / len(metrics_list),
        avg_similarity=sum(metrics.avg_similarity for metrics in metrics_list) / len(metrics_list),
    )


def _summarize_performance(results: dict) -> PerformanceMetrics:
    """Average end-to-end timings into the report-level performance section."""
    if not results:
        return PerformanceMetrics()

    count = len(results)
    retrieval_latency = sum(item["retrieval_ms"] for item in results.values()) / count
    llm_latency = sum(item["llm_ms"] for item in results.values()) / count
    total_latency = sum(item["total_ms"] for item in results.values()) / count
    return PerformanceMetrics(
        retrieval_latency_ms=retrieval_latency,
        llm_latency_ms=llm_latency,
        total_latency_ms=total_latency,
        queries_per_second=1000 / total_latency if total_latency > 0 else 0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST DATASET: Ground truth clinical cases
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "name": "Acute Kidney Injury (Critical)",
        "patient_text": """
        Creatinine: 6.2 mg/dL (normal: 0.7-1.3)
        BUN: 120 mg/dL (normal: 7-20)
        Potassium: 6.8 mEq/L (normal: 3.5-5.0)
        Urine Output: 150 mL/day (oliguric)
        Phosphate: 8.2 mg/dL (normal: 2.5-4.5)
        """,
        "expected_level": "Critical",
        "expected_score_range": (8, 10),
        "expected_sources": {"kidney_disease_guidelines.txt"},
    },
    {
        "name": "Liver Failure (High)",
        "patient_text": """
        ALT: 450 U/L (normal: 7-35)
        AST: 520 U/L (normal: 10-40)
        Bilirubin: 8.5 mg/dL (normal: 0.1-1.2)
        INR: 6.8 (normal: 0.8-1.1)
        Albumin: 1.8 g/dL (normal: 3.5-5.0)
        """,
        "expected_level": "High",
        "expected_score_range": (7, 9),
        "expected_sources": {"liver_failure_guidelines.txt"},
    },
    {
        "name": "Sepsis (High)",
        "patient_text": """
        Temperature: 39.8°C (fever)
        WBC: 18,500/µL (normal: 4,500-11,000)
        Heart Rate: 115 bpm
        Respiratory Rate: 28/min
        Lactate: 4.2 mmol/L (normal: < 2.0)
        """,
        "expected_level": "High",
        "expected_score_range": (7, 9),
        "expected_sources": {"sepsis_guidelines.txt"},
    },
    {
        "name": "Normal Labs (Low)",
        "patient_text": """
        Creatinine: 1.0 mg/dL
        BUN: 15 mg/dL
        Potassium: 4.2 mEq/L
        Glucose: 95 mg/dL
        ALT: 25 U/L
        """,
        "expected_level": "Low",
        "expected_score_range": (0, 2),
        "expected_sources": set(),
    },
    {
        "name": "Moderate Metabolic Acidosis (Moderate)",
        "patient_text": """
        pH: 7.28 (normal: 7.35-7.45)
        HCO3-: 16 mEq/L (normal: 22-26)
        PCO2: 32 mmHg
        Lactate: 2.8 mmol/L
        """,
        "expected_level": "Moderate",
        "expected_score_range": (5, 7),
        "expected_sources": {"emergency_medicine_references.txt"},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def test_retrieval_quality() -> dict:
    """
    Test retrieval quality: Do retrieved chunks match expected sources?
    
    Returns:
        Dictionary with retrieval metrics per test case.
    """
    logger.info("=" * 80)
    logger.info("TEST 1: RETRIEVAL QUALITY")
    logger.info("=" * 80)
    
    pipeline = RAGPipeline()
    results = {}
    
    for case in TEST_CASES:
        logger.info(f"\nCase: {case['name']}")
        logger.info(f"  Query: {case['patient_text'].strip()}")
        
        # Retrieve
        start = time.perf_counter()
        retrieved = pipeline.run_query(case["patient_text"], top_k=5)
        retrieval_time = (time.perf_counter() - start) * 1000
        
        # Evaluate
        retrieved_sources = {chunk.source for chunk in retrieved}
        expected_sources = case.get("expected_sources", set())
        normalized_expected = {normalize_source_name(source) for source in expected_sources}
        normalized_retrieved = {normalize_source_name(source) for source in retrieved_sources}
        
        metrics = evaluate_retrieval(retrieved, expected_sources, k_values=[1, 3, 5])
        
        # Log
        logger.info(f"  Expected sources: {expected_sources}")
        logger.info(f"  Retrieved sources: {retrieved_sources}")
        logger.info(f"  Normalized expected: {normalized_expected}")
        logger.info(f"  Normalized retrieved: {normalized_retrieved}")
        logger.info(f"  Retrieved chunks: {[chunk.short_repr() for chunk in retrieved]}")
        logger.info(f"  Chunk scores: {[round(chunk.similarity, 4) for chunk in retrieved]}")
        for chunk in retrieved[:3]:
            logger.info(f"    • {chunk.source}: {chunk.similarity:.2%} similarity")
        logger.info(f"  P@5: {metrics.precision_at_k[5]:.2f}, R@5: {metrics.recall_at_k[5]:.2f}, NDCG@5: {metrics.ndcg_at_k[5]:.2f}")
        logger.info(f"  Retrieval time: {retrieval_time:.0f} ms")
        
        results[case["name"]] = {
            "metrics": metrics,
            "latency_ms": retrieval_time,
        }
    
    return results


def test_llm_output_quality() -> dict:
    """
    Test LLM output JSON validity and schema compliance.
    
    Returns:
        Dictionary with LLM output metrics per test case.
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: LLM OUTPUT QUALITY")
    logger.info("=" * 80)
    
    pipeline = RAGPipeline()
    analyzer = SeverityAnalyzer()
    results = {}
    
    for case in TEST_CASES:
        logger.info(f"\nCase: {case['name']}")
        
        # Retrieve context first
        retrieved = pipeline.run_query(case["patient_text"], top_k=5)
        
        # Analyze
        start = time.perf_counter()
        result = analyzer.analyze(case["patient_text"], retrieved)
        llm_time = (time.perf_counter() - start) * 1000
        
        # Validate
        is_valid = result.is_valid
        in_range = 0 <= result.severity_score <= 10
        level_valid = result.severity_level in {"Low", "Moderate", "High", "Critical"}
        
        logger.info(f"  Valid JSON: {is_valid}")
        logger.info(f"  Score: {result.severity_score} (valid range: {in_range})")
        logger.info(f"  Level: {result.severity_level} (valid: {level_valid})")
        logger.info(f"  LLM time: {llm_time:.0f} ms")
        
        if result.parse_error:
            logger.warning(f"  ⚠ Parse error: {result.parse_error}")
        
        results[case["name"]] = {
            "result": result,
            "valid_json": is_valid,
            "score_in_range": in_range,
            "level_valid": level_valid,
            "latency_ms": llm_time,
        }
    
    return results


def test_clinical_accuracy() -> dict:
    """
    Test clinical accuracy: Do predictions match ground truth?
    
    Returns:
        Dictionary with accuracy metrics.
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: CLINICAL ACCURACY")
    logger.info("=" * 80)
    
    pipeline = RAGPipeline()
    analyzer = SeverityAnalyzer()
    predictions = []
    ground_truth = []
    pred_scores = []
    gt_scores = []
    
    for case in TEST_CASES:
        logger.info(f"\nCase: {case['name']}")
        
        # Retrieve context
        retrieved = pipeline.run_query(case["patient_text"], top_k=5)
        
        # Predict
        result = analyzer.analyze(case["patient_text"], retrieved)
        
        # Extract
        predictions.append(result.severity_level)
        ground_truth.append(case["expected_level"])
        pred_scores.append(result.severity_score)
        gt_scores.append(sum(case["expected_score_range"]) // 2)  # Use midpoint
        
        # Log
        expected = case["expected_level"]
        predicted = result.severity_level
        match = "✓" if predicted == expected else "✗"
        logger.info(f"  {match} Expected: {expected}, Predicted: {predicted}")
        logger.info(f"     Expected score range: {case['expected_score_range']}, Predicted: {result.severity_score}")
    
    # Compute metrics
    metrics = evaluate_clinical_accuracy(predictions, ground_truth, pred_scores, gt_scores)
    
    logger.info(f"\nOverall Accuracy: {metrics.accuracy:.1%}")
    logger.info(f"Score MAE: {metrics.score_mae:.2f}")
    logger.info(f"Score RMSE: {metrics.score_rmse:.2f}")
    for level in ["Low", "Moderate", "High", "Critical"]:
        if level in metrics.precision:
            logger.info(f"  {level}: Precision={metrics.precision[level]:.2f}, Recall={metrics.recall[level]:.2f}, F1={metrics.f1[level]:.2f}")
    
    return {"metrics": metrics}


def test_consistency() -> dict:
    """
    Test output consistency: Same input → similar outputs?
    
    Returns:
        Dictionary with consistency metrics.
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: CONSISTENCY (3 runs per query)")
    logger.info("=" * 80)
    
    pipeline = RAGPipeline()
    analyzer = SeverityAnalyzer()
    n_runs = 3
    
    all_results = []
    
    for run in range(n_runs):
        logger.info(f"\nRun {run + 1}/{n_runs}:")
        run_results = []
        
        for case in TEST_CASES[:2]:  # Only test 2 cases for speed
            retrieved = pipeline.run_query(case["patient_text"], top_k=5)
            result = analyzer.analyze(case["patient_text"], retrieved)
            run_results.append(result)
            logger.info(f"  {case['name']}: {result.severity_level} (score: {result.severity_score})")
        
        all_results.append(run_results)
    
    # Compute consistency metrics
    from evaluation.metrics import evaluate_consistency
    metrics = evaluate_consistency(all_results)
    
    logger.info(f"\nLevel Consistency: {metrics.level_consistency:.1%}")
    logger.info(f"Score Std Dev: {metrics.score_std_dev:.2f}")
    
    return {"metrics": metrics}


def test_end_to_end() -> dict:
    """
    Full end-to-end test: Retrieval → LLM → Output.
    
    Returns:
        Dictionary with full pipeline metrics.
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: END-TO-END PIPELINE")
    logger.info("=" * 80)
    
    pipeline = RAGPipeline()
    analyzer = SeverityAnalyzer()
    
    results = {}
    
    for case in TEST_CASES:
        logger.info(f"\nCase: {case['name']}")
        
        # Full pipeline
        start = time.perf_counter()
        retrieved = pipeline.run_query(case["patient_text"], top_k=5)
        retrieval_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        result = analyzer.analyze(case["patient_text"], retrieved)
        llm_time = (time.perf_counter() - start) * 1000
        
        total_time = retrieval_time + llm_time
        
        logger.info(f"  Retrieval: {retrieval_time:.0f} ms")
        logger.info(f"  LLM: {llm_time:.0f} ms")
        logger.info(f"  Total: {total_time:.0f} ms")
        logger.info(f"  Result: {result.severity_level} (score: {result.severity_score}/10)")
        
        results[case["name"]] = {
            "retrieval_ms": retrieval_time,
            "llm_ms": llm_time,
            "total_ms": total_time,
            "result": result,
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def run_all_tests() -> EvaluationReport:
    """Run all evaluation tests and generate comprehensive report."""
    
    logger.info("\n\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "CLINICAL SEVERITY RAG - AUTOMATED EVALUATION SUITE".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    
    # Run tests
    retrieval_results = test_retrieval_quality()
    llm_results = test_llm_output_quality()
    clinical_results = test_clinical_accuracy()
    consistency_results = test_consistency()
    e2e_results = test_end_to_end()
    
    # Extract metrics
    llm_results_list = [r["result"] for r in llm_results.values()]
    retrieval_metrics = _average_retrieval_metrics(retrieval_results)
    llm_metrics = evaluate_llm_output_batch(llm_results_list)
    clinical_metrics = clinical_results["metrics"]
    consistency_metrics = consistency_results["metrics"]
    performance_metrics = _summarize_performance(e2e_results)
    
    # Create report
    report = EvaluationReport(
        retrieval=retrieval_metrics,
        llm_output=llm_metrics,
        clinical_accuracy=clinical_metrics,
        performance=performance_metrics,
        consistency=consistency_metrics,
    )
    
    logger.info("\n" + str(report))
    
    # Save report as JSON
    report_file = Path("evaluation/reports") / f"report_{int(time.time())}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    
    logger.info(f"\nReport saved to: {report_file}")
    
    return report


if __name__ == "__main__":
    run_all_tests()
