"""
evaluation/example_usage.py
────────────────────────────
Simple examples of how to use evaluation metrics.

Run:
    python -m evaluation.example_usage
"""
from __future__ import annotations

import json
from pathlib import Path

from rag.rag_pipeline import RAGPipeline
from llm.severity_analyzer import SeverityAnalyzer
from evaluation.metrics import (
    evaluate_retrieval,
    evaluate_llm_output_batch,
    evaluate_clinical_accuracy,
    EvaluationReport,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Evaluate a Single Query
# ═══════════════════════════════════════════════════════════════════════════════


def example_1_single_query():
    """Evaluate retrieval and LLM for a single clinical case."""
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: Evaluate Single Query")
    logger.info("=" * 80)
    
    patient_text = """
    Creatinine: 5.8 mg/dL (normal: 0.7-1.3)
    BUN: 115 mg/dL (normal: 7-20)
    Potassium: 6.5 mEq/L (normal: 3.5-5.0)
    Urine Output: 180 mL/day
    """
    
    expected_level = "High"
    expected_sources = {"kidney_disease_guidelines.txt"}
    
    # Initialize
    pipeline = RAGPipeline()
    analyzer = SeverityAnalyzer()
    
    # Step 1: Retrieve
    logger.info("\n1. RETRIEVAL")
    retrieved = pipeline.run_query(patient_text, top_k=5)
    retrieval_metrics = evaluate_retrieval(retrieved, expected_sources, k_values=[1, 3, 5])
    
    logger.info(f"Retrieved {len(retrieved)} chunks:")
    for i, chunk in enumerate(retrieved[:3], 1):
        logger.info(f"  {i}. [{chunk.source}] {chunk.similarity:.1%} → {chunk.text[:60]}...")
    logger.info(f"\n{retrieval_metrics}")
    
    # Step 2: Analyze
    logger.info("\n2. LLM ANALYSIS")
    result = analyzer.analyze(patient_text, retrieved)
    
    logger.info(f"Severity Level: {result.severity_level}")
    logger.info(f"Severity Score: {result.severity_score}/10")
    logger.info(f"Valid: {result.is_valid}")
    logger.info(f"Findings: {', '.join(result.key_findings[:2])}")
    logger.info(f"Evidence: {', '.join(result.evidence[:2])}")
    
    # Step 3: Compare
    logger.info("\n3. COMPARISON")
    logger.info(f"Expected: {expected_level}, Got: {result.severity_level}")
    logger.info(f"Match: {'✓' if result.severity_level == expected_level else '✗'}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Batch Evaluation of Multiple Cases
# ═══════════════════════════════════════════════════════════════════════════════


def example_2_batch_evaluation():
    """Evaluate multiple test cases and compute aggregate metrics."""
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Batch Evaluation")
    logger.info("=" * 80)
    
    test_cases = [
        {
            "name": "AKI",
            "text": "Creatinine 6.2, BUN 120, K 6.8, Urine 150 mL/day",
            "expected": "High",
        },
        {
            "name": "Liver Failure",
            "text": "ALT 450, AST 520, Bilirubin 8.5, INR 6.8",
            "expected": "High",
        },
        {
            "name": "Normal",
            "text": "Creatinine 1.0, BUN 15, K 4.2, Glucose 95",
            "expected": "Low",
        },
    ]
    
    pipeline = RAGPipeline()
    analyzer = SeverityAnalyzer()
    predictions = []
    ground_truth = []
    results_list = []
    
    for case in test_cases:
        logger.info(f"\n{case['name']}: {case['text'][:50]}...")
        
        retrieved = pipeline.run_query(case["text"], top_k=5)
        result = analyzer.analyze(case["text"], retrieved)
        predictions.append(result.severity_level)
        ground_truth.append(case["expected"])
        results_list.append(result)
        
        match = "✓" if result.severity_level == case["expected"] else "✗"
        logger.info(f"  {match} Expected: {case['expected']}, Got: {result.severity_level}")
    
    # Compute metrics
    logger.info("\nAGGREGATE METRICS")
    llm_metrics = evaluate_llm_output_batch(results_list)
    accuracy_metrics = evaluate_clinical_accuracy(predictions, ground_truth)
    
    logger.info(f"\n{llm_metrics}")
    logger.info(f"\n{accuracy_metrics}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Track Metrics Over Time
# ═══════════════════════════════════════════════════════════════════════════════


def example_3_metrics_tracking():
    """Run evaluation and save results for tracking performance over time."""
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Metrics Tracking")
    logger.info("=" * 80)
    
    analyzer = SeverityAnalyzer()
    
    test_cases = [
        "Creatinine 6.2, BUN 120, K 6.8",  # Expected: Critical
        "ALT 450, AST 520, Bilirubin 8.5",  # Expected: High
        "Creatinine 1.0, BUN 15, K 4.2",    # Expected: Low
    ]
    
    results_list = []
    for text in test_cases:
        result = analyzer.analyze(text)
        results_list.append(result)
    
    # Create report
    report = EvaluationReport(
        llm_output=evaluate_llm_output_batch(results_list),
    )
    
    # Save to JSON
    report_file = Path("evaluation/reports/example_tracking.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    
    logger.info(f"Report saved to {report_file}")
    logger.info(f"\n{report}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Custom Evaluation Workflow
# ═══════════════════════════════════════════════════════════════════════════════


def example_4_custom_workflow():
    """Custom evaluation workflow tailored to your needs."""
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Custom Evaluation Workflow")
    logger.info("=" * 80)
    
    pipeline = RAGPipeline()
    analyzer = SeverityAnalyzer()
    
    # Your critical test case
    patient_text = """
    Chief Complaint: Kidney dysfunction
    
    Labs:
    - Creatinine: 4.5 mg/dL
    - BUN: 85 mg/dL
    - Potassium: 6.2 mEq/L
    - Urine output: 300 mL/day
    
    Clinical: Oliguric AKI
    """
    
    logger.info("STEP 1: Quality Checks")
    
    # Check 1: Knowledge base ready?
    kb_ready = pipeline.is_knowledge_base_ready()
    logger.info(f"  KB Ready: {kb_ready}")
    
    # Check 2: Retrieve relevant guidelines
    retrieved = pipeline.run_query(patient_text, top_k=3)
    logger.info(f"  Retrieved: {len(retrieved)} chunks")
    
    for chunk in retrieved:
        logger.info(f"    • [{chunk.source}] {chunk.similarity:.0%}")
    
    logger.info("\nSTEP 2: Generate Assessment")
    
    # Generate result
    result = analyzer.analyze(patient_text)
    
    # Validate
    is_critical = result.severity_level == "Critical"
    high_score = result.severity_score >= 8
    has_findings = len(result.key_findings) > 0
    has_evidence = len(result.evidence) > 0
    
    logger.info(f"  Severity Level: {result.severity_level} {'✓' if is_critical else '✗'}")
    logger.info(f"  Severity Score: {result.severity_score}/10 {'✓' if high_score else '✗'}")
    logger.info(f"  Has Findings: {len(result.key_findings)} {'✓' if has_findings else '✗'}")
    logger.info(f"  Has Evidence: {len(result.evidence)} {'✓' if has_evidence else '✗'}")
    
    logger.info("\nSTEP 3: Final Assessment")
    
    all_checks = is_critical and high_score and has_findings and has_evidence
    status = "✓ PASS" if all_checks else "✗ FAIL"
    logger.info(f"  Overall: {status}")
    
    if result.summary:
        logger.info(f"  Summary: {result.summary[:100]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    logger.info("\n\nCLINICAL SEVERITY RAG - EVALUATION EXAMPLES\n")
    
    example_1_single_query()
    example_2_batch_evaluation()
    example_3_metrics_tracking()
    example_4_custom_workflow()
    
    logger.info("\n\n" + "=" * 80)
    logger.info("Examples complete! See evaluation/README.md for more details.")
    logger.info("=" * 80 + "\n")
