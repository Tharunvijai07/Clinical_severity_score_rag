# Evaluation Metrics Guide

## Overview

This folder contains **automated evaluation metrics** for the Clinical Severity RAG system. Instead of manual testing, you can run objective measurements across 5 dimensions:

| Metric Category | What It Measures | Key Metrics |
|---|---|---|
| **Retrieval Quality** | Are the right medical guidelines retrieved? | Precision@K, Recall@K, NDCG@K, MRR |
| **LLM Output Quality** | Is the JSON valid and schema-compliant? | JSON validity %, schema compliance %, parse errors |
| **Clinical Accuracy** | Do predictions match ground truth? | Accuracy, MAE, RMSE, F1-score per class |
| **Performance** | How fast is the system? | Retrieval latency, LLM latency, throughput |
| **Consistency** | Does same input produce same output? | Level consistency %, score std dev |

---

## Quick Start

### 1. Run Full Evaluation Suite

```bash
python -m evaluation.test_suite
```

This runs 5 tests with 5 test cases each and generates a comprehensive report.

**Output:**
- Console logs with detailed results
- JSON report saved to `evaluation/reports/report_<timestamp>.json`

### 2. Run Specific Tests Individually

```python
from evaluation.test_suite import (
    test_retrieval_quality,
    test_llm_output_quality,
    test_clinical_accuracy,
    test_consistency,
    test_end_to_end,
)

# Test 1: Retrieval
retrieval_results = test_retrieval_quality()

# Test 2: LLM Output
llm_results = test_llm_output_quality()

# Test 3: Clinical Accuracy
accuracy_results = test_clinical_accuracy()

# Test 4: Consistency
consistency_results = test_consistency()

# Test 5: End-to-End
e2e_results = test_end_to_end()
```

### 3. Build Custom Evaluation

```python
from evaluation.metrics import (
    evaluate_retrieval,
    evaluate_llm_output_batch,
    evaluate_clinical_accuracy,
    EvaluationReport,
)
from rag.rag_pipeline import RAGPipeline
from llm.severity_analyzer import SeverityAnalyzer

# Your data
patient_text = "Creatinine 6.2, BUN 120, K 6.8..."
expected_level = "Critical"
expected_sources = {"kidney_disease_guidelines.txt"}

# Step 1: Evaluate Retrieval
pipeline = RAGPipeline()
retrieved = pipeline.run_query(patient_text, top_k=5)
retrieval_metrics = evaluate_retrieval(retrieved, expected_sources)
print(retrieval_metrics)

# Step 2: Evaluate LLM Output
analyzer = SeverityAnalyzer()
result = analyzer.analyze(patient_text)
llm_metrics = evaluate_llm_output_batch([result])
print(llm_metrics)

# Step 3: Evaluate Clinical Accuracy
accuracy_metrics = evaluate_clinical_accuracy(
    predictions=[result.severity_level],
    ground_truth=[expected_level]
)
print(accuracy_metrics)

# Generate Report
report = EvaluationReport(
    retrieval=retrieval_metrics,
    llm_output=llm_metrics,
    clinical_accuracy=accuracy_metrics,
)
print(report)
```

---

## Metrics Explained

### Retrieval Metrics

**Precision @ K (P@K)**
- What % of top-K results are relevant?
- Formula: `(# relevant in top K) / K`
- Range: [0, 1]
- Good target: P@5 ≥ 0.6

**Recall @ K (R@K)**
- What % of all relevant items appear in top-K?
- Formula: `(# relevant in top K) / (total # relevant)`
- Range: [0, 1]
- Good target: R@5 ≥ 0.8

**NDCG @ K (Normalized Discounted Cumulative Gain)**
- Rewards ranking relevant items higher
- Range: [0, 1]
- Good target: NDCG@5 ≥ 0.7

**Mean Reciprocal Rank (MRR)**
- Position of first relevant item: `1 / rank`
- Range: [0, 1]
- Good target: MRR ≥ 0.5

**Average Similarity**
- Mean cosine similarity of retrieved chunks
- Range: [0, 1]
- Good target: ≥ 0.5

---

### LLM Output Metrics

**JSON Validity Rate**
- % of outputs that parse without JSON errors
- Target: 100%

**Schema Compliance Rate**
- % of outputs with correct JSON structure
- Target: 100%

**Score Range Compliance**
- % of severity_score values in [0, 10]
- Target: 100%

**Level Compliance**
- % of severity_level in {Low, Moderate, High, Critical}
- Target: 100%

**Average Parse Errors**
- Number of validation errors per output
- Target: 0.0

---

### Clinical Accuracy Metrics

**Accuracy**
- % of predictions matching ground truth
- Formula: `(# correct) / (# total)`
- Target: ≥ 90% for well-calibrated model

**Score MAE (Mean Absolute Error)**
- Average absolute difference in severity score
- Target: ≤ 1.5 points

**Score RMSE (Root Mean Square Error)**
- Penalizes larger errors more heavily
- Target: ≤ 2.0 points

**Precision, Recall, F1 (Per Severity Level)**
- Per-class breakdown of accuracy
- Target: All ≥ 0.8

**Confusion Matrix**
- Breakdown of predictions vs ground truth
- Shows which misclassifications occur

---

### Performance Metrics

**Retrieval Latency**
- Time to retrieve top-K chunks
- Target: < 500 ms

**LLM Latency**
- Time for Gemini/Groq inference
- Target: < 5 seconds

**Total Latency**
- End-to-end time
- Target: < 6 seconds

**Queries Per Second**
- Throughput
- Target: > 0.1 q/s (10 sec per query is OK)

---

### Consistency Metrics

**Level Consistency**
- % of runs with same severity level
- Tested across 3 identical queries
- Target: ≥ 90%

**Score Std Dev**
- Standard deviation of severity scores
- Lower is better (consistent)
- Target: ≤ 0.5

---

## Test Dataset

Default test cases in `test_suite.py`:

1. **Acute Kidney Injury (Critical)** - Creatinine 6.2, BUN 120, K 6.8
2. **Liver Failure (High)** - ALT 450, AST 520, Bilirubin 8.5
3. **Sepsis (High)** - Fever 39.8°C, WBC 18.5k, Lactate 4.2
4. **Normal Labs (Low)** - All values in normal range
5. **Moderate Metabolic Acidosis (Moderate)** - pH 7.28, HCO3 16

### Add Custom Test Cases

Edit `evaluation/test_suite.py` and add to `TEST_CASES`:

```python
{
    "name": "Your Test Case",
    "patient_text": """
    Lab values here...
    """,
    "expected_level": "High",  # or Low/Moderate/Critical
    "expected_score_range": (7, 9),
    "expected_sources": {"relevant_guideline.txt"},
}
```

---

## Interpreting Results

### Good Performance ✓
- Retrieval: P@5 ≥ 0.6, NDCG@5 ≥ 0.7
- LLM Output: 100% JSON validity & schema compliance
- Clinical Accuracy: ≥ 90%
- Performance: < 6 seconds total
- Consistency: ≥ 90% level consistency

### Needs Improvement ⚠
- Retrieval: P@5 < 0.6 → Adjust chunk size, embedding model, or add more KB
- LLM Output: Parse errors > 0 → Check prompt template or model response
- Clinical Accuracy: < 85% → Verify test cases, adjust prompt, or collect more training data
- Performance: > 10 seconds → Use GPU, optimize retrieval, or batch queries
- Consistency: < 80% → May indicate temperature too high or model instability

---

## Saving Results

Reports are automatically saved to `evaluation/reports/report_<timestamp>.json`:

```json
{
  "timestamp": "2026-08-16 14:30:45",
  "retrieval": {
    "precision_at_k": {"1": 0.8, "3": 0.67, "5": 0.6},
    "recall_at_k": {"1": 0.2, "3": 0.5, "5": 0.8},
    ...
  },
  "llm_output": {
    "json_validity_rate": 1.0,
    ...
  },
  ...
}
```

Load and analyze:

```python
import json

with open("evaluation/reports/report_1692181445.json") as f:
    report = json.load(f)

print(f"Accuracy: {report['clinical_accuracy']['accuracy']:.1%}")
print(f"P@5: {report['retrieval']['precision_at_k']['5']:.2f}")
```

---

## Advanced: Custom Metrics

Create your own metric function:

```python
# evaluation/custom_metrics.py
from evaluation.metrics import RetrievedChunk

def my_custom_metric(retrieved: list[RetrievedChunk]) -> float:
    """Calculate custom metric from retrieved chunks."""
    # Your implementation
    return score

# Use it
from evaluation.custom_metrics import my_custom_metric
score = my_custom_metric(retrieved_chunks)
```

---

## Troubleshooting

**Q: Test fails with "RAG pipeline not ready"**  
A: Run `python setup_knowledge_base.py` first to build the vector store.

**Q: LLM metrics show 0% validity**  
A: Check GROQ_API_KEY environment variable and model output format.

**Q: Retrieval shows 0% precision**  
A: Verify ground truth sources match file names in `knowledge_base/`.

**Q: Performance metrics are slow**  
A: First run embeds models locally (can take 1-2 min). Subsequent runs are faster.

---

## Next Steps

1. **Run the suite**: `python -m evaluation.test_suite`
2. **Review results**: Check console output and JSON report
3. **Add test cases**: Customize with your clinical scenarios
4. **Track over time**: Run periodically to detect regressions
5. **Iterate**: Use metrics to guide improvements
