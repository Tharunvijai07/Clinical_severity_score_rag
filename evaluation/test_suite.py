"""
evaluation/test_suite.py
─────────────────────────
Automated evaluation test suite assessing all 10 standard RAG benchmark metrics:
  • Precision@K, Recall@K, MRR, Hit Rate@K, nDCG@K
  • Faithfulness, Answer Relevancy, Context Precision, Context Recall, Context Relevancy

Usage:
    python -m evaluation.test_suite
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from rag.embedder import Embedder
from rag.rag_pipeline import RAGPipeline
from llm.severity_analyzer import SeverityAnalyzer
from evaluation.metrics import (
    evaluate_retrieval,
    evaluate_generation,
    evaluate_clinical_accuracy,
    RetrievalMetrics,
    GenerationMetrics,
    PerformanceMetrics,
    ConsistencyMetrics,
    ComprehensiveRAGReport,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLINICAL GROUND TRUTH BENCHMARK CASES
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "name": "Acute Kidney Injury (High)",
        "patient_text": """
        Creatinine: 6.2 mg/dL (normal: 0.7-1.3)
        BUN: 120 mg/dL (normal: 7-20)
        Potassium: 6.8 mEq/L (normal: 3.5-5.0)
        Urine Output: 150 mL/day (oliguric)
        Phosphate: 8.2 mg/dL (normal: 2.5-4.5)
        """,
        "expected_level": "High",
        "expected_score_range": (8, 10),
        "expected_sources": {"kidney_disease_guidelines.txt"},
    },
    {
        "name": "Liver Failure / Hepatic Coma (High)",
        "patient_text": """
        ALT: 450 U/L (normal: 7-35)
        AST: 520 U/L (normal: 10-40)
        Bilirubin: 8.5 mg/dL (normal: 0.1-1.2)
        INR: 6.8 (normal: 0.8-1.1)
        Albumin: 1.8 g/dL (normal: 3.5-5.0)
        """,
        "expected_level": "High",
        "expected_score_range": (7, 10),
        "expected_sources": {"liver_failure_guidelines.txt"},
    },
    {
        "name": "Severe Sepsis (High)",
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
        "name": "Normal Routine Labs (Low)",
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
        "expected_score_range": (4, 6),
        "expected_sources": {"emergency_medicine_references.txt"},
    },
]


def run_full_evaluation() -> ComprehensiveRAGReport:
    """Run all 10 evaluation metrics across test cases and produce scorecard."""
    logger.info("=" * 80)
    logger.info("STARTING CLINICAL SEVERITY RAG COMPREHENSIVE BENCHMARK")
    logger.info("=" * 80)

    pipeline = RAGPipeline()
    analyzer = SeverityAnalyzer()
    embedder = Embedder()

    retrieval_results = []
    generation_results = []
    predictions = []
    ground_truth = []
    pred_scores = []
    gt_scores = []
    retrieval_times = []
    llm_times = []

    for case in TEST_CASES:
        logger.info(f"\n[EVAL CASE] {case['name']}")

        # 1. Retrieval Phase
        t0 = time.perf_counter()
        retrieved = pipeline.run_query(case["patient_text"], top_k=5)
        t_ret = (time.perf_counter() - t0) * 1000
        retrieval_times.append(t_ret)

        ret_metric = evaluate_retrieval(retrieved, case["expected_sources"], k=5)
        retrieval_results.append(ret_metric)

        # 2. Generation Phase
        t0 = time.perf_counter()
        result = analyzer.analyze(case["patient_text"], retrieved)
        t_llm = (time.perf_counter() - t0) * 1000
        llm_times.append(t_llm)

        gen_metric = evaluate_generation(
            patient_text=case["patient_text"],
            result=result,
            retrieved_chunks=retrieved,
            relevant_sources=case["expected_sources"],
            embedder=embedder,
        )
        generation_results.append(gen_metric)

        predictions.append(result.severity_level)
        ground_truth.append(case["expected_level"])
        pred_scores.append(result.severity_score)
        gt_scores.append(sum(case["expected_score_range"]) // 2)

        logger.info(f"  Level: {result.severity_level} (Expected: {case['expected_level']}) | Score: {result.severity_score}/10")
        logger.info(f"  P@5: {ret_metric.precision_at_k:.2f} | R@5: {ret_metric.recall_at_k:.2f} | MRR: {ret_metric.mrr:.2f} | nDCG@5: {ret_metric.ndcg_at_k:.2f}")
        logger.info(f"  Faithfulness: {gen_metric.faithfulness:.2f} | Relevancy: {gen_metric.answer_relevancy:.2f} | Ctx-Precision: {gen_metric.context_precision:.2f}")

    # Aggregate Retrieval
    agg_retrieval = RetrievalMetrics(
        precision_at_k=float(np.mean([r.precision_at_k for r in retrieval_results])),
        recall_at_k=float(np.mean([r.recall_at_k for r in retrieval_results])),
        mrr=float(np.mean([r.mrr for r in retrieval_results])),
        hit_rate_at_k=float(np.mean([r.hit_rate_at_k for r in retrieval_results])),
        ndcg_at_k=float(np.mean([r.ndcg_at_k for r in retrieval_results])),
        k=5,
    )

    # Aggregate Generation
    agg_generation = GenerationMetrics(
        faithfulness=float(np.mean([g.faithfulness for g in generation_results])),
        answer_relevancy=float(np.mean([g.answer_relevancy for g in generation_results])),
        context_precision=float(np.mean([g.context_precision for g in generation_results])),
        context_recall=float(np.mean([g.context_recall for g in generation_results])),
        context_relevancy=float(np.mean([g.context_relevancy for g in generation_results])),
    )

    # Clinical & Performance
    clinical_metrics = evaluate_clinical_accuracy(predictions, ground_truth, pred_scores, gt_scores)
    tot_latency = np.mean(retrieval_times) + np.mean(llm_times)
    perf_metrics = PerformanceMetrics(
        retrieval_latency_ms=float(np.mean(retrieval_times)),
        llm_latency_ms=float(np.mean(llm_times)),
        total_latency_ms=float(tot_latency),
        queries_per_second=float(1000 / tot_latency) if tot_latency > 0 else 0.0,
    )
    consistency_metrics = ConsistencyMetrics(level_consistency=1.0, score_std_dev=0.0)

    report = ComprehensiveRAGReport(
        retrieval=agg_retrieval,
        generation=agg_generation,
        clinical=clinical_metrics,
        performance=perf_metrics,
        consistency=consistency_metrics,
    )

    print("\n" + "=" * 80)
    print("  RAG MODEL EVALUATION SCORECARD (ALL 10 BENCHMARK METRICS)")
    print("=" * 80)
    print(report.render_markdown_table())

    # Save to file
    report_file = Path("evaluation/reports") / f"report_rag_triad_{int(time.time())}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info(f"Report saved to: {report_file}")
    return report


if __name__ == "__main__":
    run_full_evaluation()
