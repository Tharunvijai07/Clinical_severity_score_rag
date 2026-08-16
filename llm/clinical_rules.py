"""
Rule-based clinical anchors for severity scoring.

These anchors give the LLM a calibrated baseline from objective lab and vital
abnormalities without replacing the final clinical judgement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ClinicalAnchorAssessment:
    """Structured summary of rule-based severity signals."""

    score: int
    level: str
    findings: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        if self.findings:
            findings = "\n".join(f"- {finding}" for finding in self.findings)
        else:
            findings = "- No strong rule-based severity anchors detected."

        return (
            f"Rule-based baseline severity: {self.level} ({self.score}/10)\n"
            f"Detected anchors:\n{findings}"
        )


_LAB_ALIASES = {
    "creatinine": ["creatinine", "cr"],
    "bun": ["bun", "blood urea nitrogen"],
    "potassium": ["potassium", "k"],
    "hemoglobin": ["hemoglobin", "hb", "hgb"],
    "platelets": ["platelets", "plt"],
    "inr": ["inr"],
    "lactate": ["lactate"],
    "ph": ["ph"],
    "hco3": ["hco3", "hco3-", "bicarbonate"],
    "wbc": ["wbc", "white blood cell"],
    "bilirubin": ["bilirubin"],
    "alt": ["alt"],
    "ast": ["ast"],
    "troponin": ["troponin"],
    "oxygen_saturation": ["oxygen saturation", "spo2", "o2 sat"],
    "systolic_bp": ["systolic bp", "sbp"],
}


def _extract_value(text: str, aliases: list[str]) -> float | None:
    for alias in aliases:
        pattern = rf"\b{re.escape(alias)}\b\s*[:=\-]*\s*([<>]?\s*\d+(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace("<", "").replace(">", "").strip())
    return None


def _level_from_score(score: int) -> str:
    if score <= 4:
        return "Low"
    if score <= 6:
        return "Moderate"
    if score <= 8:
        return "High"
    return "Critical"


def evaluate_clinical_anchors(patient_text: str) -> ClinicalAnchorAssessment:
    """Extract common clinical signals and produce a baseline severity score."""
    values = {
        name: _extract_value(patient_text, aliases)
        for name, aliases in _LAB_ALIASES.items()
    }

    score = 0
    findings: list[str] = []

    def add(points: int, finding: str) -> None:
        nonlocal score
        score += points
        findings.append(f"{finding} (+{points})")

    creatinine = values["creatinine"]
    if creatinine is not None:
        if creatinine > 5:
            add(3, f"Creatinine {creatinine:g} suggests severe kidney injury")
        elif creatinine >= 2:
            add(1, f"Creatinine {creatinine:g} is moderately elevated")

    bun = values["bun"]
    if bun is not None:
        if bun > 100:
            add(2, f"BUN {bun:g} is critically elevated")
        elif bun >= 40:
            add(1, f"BUN {bun:g} is elevated")

    potassium = values["potassium"]
    if potassium is not None:
        if potassium > 6:
            add(2, f"Potassium {potassium:g} is dangerous hyperkalemia")
        elif potassium >= 5.5:
            add(1, f"Potassium {potassium:g} is mild-moderate hyperkalemia")

    hemoglobin = values["hemoglobin"]
    if hemoglobin is not None:
        if hemoglobin < 6:
            add(2, f"Hemoglobin {hemoglobin:g} indicates severe anemia")
        elif hemoglobin < 8:
            add(1, f"Hemoglobin {hemoglobin:g} indicates significant anemia")

    platelets = values["platelets"]
    if platelets is not None:
        if platelets < 20000:
            add(2, f"Platelets {platelets:g} indicate severe thrombocytopenia")
        elif platelets < 50000:
            add(1, f"Platelets {platelets:g} indicate thrombocytopenia")

    inr = values["inr"]
    if inr is not None:
        if inr > 5:
            add(2, f"INR {inr:g} indicates severe coagulopathy")
        elif inr >= 1.5:
            add(1, f"INR {inr:g} is elevated")

    lactate = values["lactate"]
    if lactate is not None:
        if lactate >= 4:
            add(2, f"Lactate {lactate:g} indicates high-risk hypoperfusion")
        elif lactate >= 2:
            add(1, f"Lactate {lactate:g} is elevated")

    ph = values["ph"]
    if ph is not None:
        if ph < 7.2:
            add(2, f"pH {ph:g} indicates severe acidemia")
        elif ph < 7.35:
            add(2, f"pH {ph:g} indicates mild-moderate acidemia")

    hco3 = values["hco3"]
    if hco3 is not None:
        if hco3 < 12:
            add(2, f"HCO3 {hco3:g} indicates severe metabolic acidosis")
        elif hco3 < 20:
            add(2, f"HCO3 {hco3:g} indicates metabolic acidosis")

    wbc = values["wbc"]
    if wbc is not None:
        if wbc >= 18000:
            add(1, f"WBC {wbc:g} supports significant inflammatory response")
        elif wbc < 3000:
            add(1, f"WBC {wbc:g} is low and may indicate high-risk infection")

    bilirubin = values["bilirubin"]
    if bilirubin is not None:
        if bilirubin >= 8:
            add(2, f"Bilirubin {bilirubin:g} indicates severe liver dysfunction")
        elif bilirubin >= 3:
            add(1, f"Bilirubin {bilirubin:g} is elevated")

    alt = values["alt"]
    ast = values["ast"]
    if (alt is not None and alt >= 400) or (ast is not None and ast >= 400):
        add(1, "Marked transaminase elevation suggests significant hepatocellular injury")

    troponin = values["troponin"]
    if troponin is not None and troponin > 0.04:
        add(1, f"Troponin {troponin:g} suggests myocardial injury")

    oxygen_saturation = values["oxygen_saturation"]
    if oxygen_saturation is not None:
        if oxygen_saturation < 90:
            add(2, f"Oxygen saturation {oxygen_saturation:g}% is critically low")
        elif oxygen_saturation < 94:
            add(1, f"Oxygen saturation {oxygen_saturation:g}% is low")

    systolic_bp = values["systolic_bp"]
    if systolic_bp is not None and systolic_bp < 90:
        add(2, f"Systolic BP {systolic_bp:g} suggests shock physiology")

    score = min(score, 10)
    return ClinicalAnchorAssessment(score=score, level=_level_from_score(score), findings=findings)
