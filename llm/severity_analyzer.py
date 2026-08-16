"""
llm/severity_analyzer.py
─────────────────────────
Orchestrates the full LLM-side analysis:
  1. Builds the prompt (patient text + retrieved context)
  2. Calls Gemini
  3. Parses and validates the JSON response
  4. Returns a structured SeverityResult dataclass

Schema:
    severity_score  : int       (0–10)
    severity_level  : str       (Low | Moderate | High | Critical)
    key_findings    : list[str]
    evidence        : list[str]
    summary         : str
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from llm.clinical_rules import evaluate_clinical_anchors
from llm.gemini_client import GeminiClient
from llm.prompts import SYSTEM_PROMPT, build_severity_prompt
from rag.retriever import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_LEVELS = {"Low", "Moderate", "High", "Critical"}


@dataclass
class SeverityResult:
    """Parsed clinical severity assessment returned by Gemini."""

    severity_score: int
    severity_level: str
    confidence: float
    key_findings: list[str]
    evidence: list[str]
    summary: str
    raw_response: str = field(default="", repr=False)
    parse_error: str = field(default="", repr=False)

    @property
    def is_valid(self) -> bool:
        return not bool(self.parse_error)

    @property
    def color(self) -> str:
        """Map severity level to a display colour."""
        return {
            "Low":      "#22c55e",   # green
            "Moderate": "#eab308",   # amber
            "High":     "#f97316",   # orange
            "Critical": "#ef4444",   # red
        }.get(self.severity_level, "#94a3b8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity_score": self.severity_score,
            "severity_level": self.severity_level,
            "confidence":     self.confidence,
            "key_findings":   self.key_findings,
            "evidence":       self.evidence,
            "summary":        self.summary,
        }


def _extract_json(text: str) -> str:
    """
    Try to isolate a JSON object from the model response.
    Handles cases where the model wraps it in markdown fences.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()

    # Find the first { … } block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


class SeverityAnalyzer:
    """
    End-to-end severity analysis: prompt building → Gemini → parsed result.

    Parameters
    ----------
    gemini_client : GeminiClient, optional
        Pre-built client; a default one is created if not provided.
    """

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        self._client = gemini_client or GeminiClient()

    def analyze(
        self,
        patient_text: str,
        context_chunks: list[RetrievedChunk] | None = None,
    ) -> SeverityResult:
        """
        Run the full analysis pipeline.

        Parameters
        ----------
        patient_text : str
            Extracted text from the patient lab report.
        context_chunks : list[RetrievedChunk]
            Retrieved medical guideline passages.

        Returns
        -------
        SeverityResult
            Parsed, validated severity assessment.
        """
        context_chunks = context_chunks or []
        anchors = evaluate_clinical_anchors(patient_text)

        logger.info(
            f"Starting severity analysis — "
            f"patient_text={len(patient_text):,} chars, "
            f"context_chunks={len(context_chunks)}, "
            f"anchor={anchors.level} ({anchors.score}/10)"
        )

        # 1. Build prompt
        prompt = build_severity_prompt(
            patient_text,
            context_chunks,
            clinical_anchors=anchors.to_prompt_block(),
        )

        # 2. Call Gemini
        try:
            raw = self._client.generate(
                prompt=prompt,
                system_instruction=SYSTEM_PROMPT,
            )
        except RuntimeError as exc:
            logger.error(f"Gemini call failed: {exc}")
            return self._error_result(str(exc), raw_response="")

        # 3. Parse JSON
        return self._parse_response(raw)

    # ── Parsing helpers ────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> SeverityResult:
        """Parse and validate Gemini's raw text response → SeverityResult."""
        try:
            json_str = _extract_json(raw)
            data: dict[str, Any] = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(f"JSON parse error: {exc}\nRaw:\n{raw[:500]}")
            return self._error_result(
                f"Could not parse model response as JSON: {exc}",
                raw_response=raw,
            )

        # Validate fields
        errors: list[str] = []

        score = data.get("severity_score")
        if not isinstance(score, (int, float)):
            errors.append(f"severity_score must be numeric, got: {type(score)}")
            score = 0
        score = max(0, min(10, int(score)))

        level = data.get("severity_level", "")
        if level not in _VALID_LEVELS:
            # Auto-infer from score
            logger.warning(f"Invalid severity_level '{level}', inferring from score.")
            level = self._score_to_level(score)

        confidence = data.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            errors.append(f"confidence must be numeric, got: {type(confidence)}")
            confidence = 0.0
        confidence = max(0.0, min(1.0, float(confidence)))

        findings = data.get("key_findings", [])
        if not isinstance(findings, list):
            findings = [str(findings)]

        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        summary = str(data.get("summary", "No summary provided."))

        if errors:
            logger.warning(f"Validation issues: {errors}")

        return SeverityResult(
            severity_score=score,
            severity_level=level,
            confidence=confidence,
            key_findings=[str(f) for f in findings],
            evidence=[str(e) for e in evidence],
            summary=summary,
            raw_response=raw,
            parse_error="; ".join(errors),
        )

    @staticmethod
    def _score_to_level(score: int) -> str:
        if score <= 4:
            return "Low"
        elif score <= 6:
            return "Moderate"
        elif score <= 8:
            return "High"
        return "Critical"

    @staticmethod
    def _error_result(error_msg: str, raw_response: str) -> SeverityResult:
        return SeverityResult(
            severity_score=0,
            severity_level="Low",
            confidence=0.0,
            key_findings=["Analysis failed — see error details."],
            evidence=[],
            summary=f"Error during analysis: {error_msg}",
            raw_response=raw_response,
            parse_error=error_msg,
        )
