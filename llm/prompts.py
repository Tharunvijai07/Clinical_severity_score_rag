"""
llm/prompts.py
───────────────
All prompt templates for the clinical severity RAG system.

Keeping prompts in a dedicated module makes them easy to iterate on
independently from the calling code.
"""
from __future__ import annotations

from rag.retriever import RetrievedChunk

# ── System Instruction ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MedSeverity-AI, an advanced Clinical Decision Support System
specialising in evidence-based severity assessment of patient laboratory reports.

Your core responsibilities:
1. Analyse patient laboratory findings objectively and methodically.
2. Ground every clinical judgement in the provided medical guideline excerpts.
3. Assign a severity score and level that accurately reflects the patient's condition.
4. Cite specific values and thresholds from the guidelines as evidence.
5. Never fabricate medical information not present in the provided context.

You output ONLY valid JSON — no markdown fences, no commentary outside the JSON object.
The JSON must strictly conform to the schema provided in the user prompt.
"""

# ── Severity Prompt ───────────────────────────────────────────────────────────

_SEVERITY_PROMPT_TEMPLATE = """\
=== PATIENT LABORATORY REPORT ===
{patient_text}

=== RULE-BASED CLINICAL ANCHORS ===
{clinical_anchors}

=== RETRIEVED MEDICAL GUIDELINE CONTEXT ===
{context_block}

=== TASK ===
Based ONLY on the patient findings above and the retrieved medical guideline excerpts,
produce a comprehensive clinical severity assessment.

Return a single, valid JSON object with EXACTLY these keys:

{{
  "severity_score": <integer 0–10>,
  "severity_level": "<one of: Low | Moderate | High | Critical>",
  "confidence": <number 0.0–1.0>,
  "key_findings": [
    "<finding 1>",
    "<finding 2>",
    ...
  ],
  "evidence": [
    "<specific guideline-referenced threshold or criterion>",
    ...
  ],
  "summary": "<2–4 sentence clinical narrative synthesising the findings>"
}}

Scoring guide:
  0–2  → Low      (all values within or near reference ranges, no immediate risk)
  3–4  → Low-Moderate (mild abnormalities, outpatient management)
  5–6  → Moderate  (significant abnormalities, close monitoring required)
  7–8  → High      (severe derangements, hospital admission indicated)
  9–10 → Critical  (life-threatening values, immediate intervention required)

Use Moderate severity when:
- 1-2 organ systems are mildly or moderately abnormal.
- There is no clear organ failure, shock, or immediate life-threatening abnormality.
- The patient requires monitoring, repeat labs, or urgent follow-up but not ICU-level intervention.

Use High severity when:
- There are significant abnormalities in multiple systems.
- The patient has high risk of deterioration or likely needs hospital admission.

Use Critical severity when:
- There are immediate life-threatening abnormalities.
- There is organ failure, shock physiology, severe hypoxemia, severe acidosis, or dangerous electrolyte derangement.

Rules:
- severity_score must be an integer (not a float).
- severity_level must exactly match one of the four options listed.
- confidence must be a number between 0.0 and 1.0.
- key_findings must be a JSON array of strings (minimum 1, maximum 10).
- evidence must be a JSON array of strings citing specific thresholds from the context.
- summary must be a single string (no embedded newlines).
- If a finding is ambiguous without full clinical context, note the limitation in summary.
- Do NOT include any text outside the JSON object.
"""


def build_severity_prompt(
    patient_text: str,
    context_chunks: list[RetrievedChunk],
    clinical_anchors: str = "No rule-based clinical anchors provided.",
) -> str:
    """
    Build the final user-turn prompt from patient text and retrieved chunks.

    Parameters
    ----------
    patient_text : str
        Extracted text from the uploaded patient PDF.
    context_chunks : list[RetrievedChunk]
        Retrieved medical guideline passages from ChromaDB.

    Returns
    -------
    str
        Formatted prompt ready for the configured LLM.
    """
    if context_chunks:
        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, start=1):
            formatted_chunks.append(
                f"[Guideline Excerpt {i} — Source: {chunk.source}]\n{chunk.text}"
            )
        context_block = "\n\n".join(formatted_chunks)
    else:
        context_block = (
            "No specific guideline excerpts retrieved. "
            "Base assessment on general clinical principles only."
        )

    return _SEVERITY_PROMPT_TEMPLATE.format(
        patient_text=patient_text.strip(),
        clinical_anchors=clinical_anchors,
        context_block=context_block,
    )
