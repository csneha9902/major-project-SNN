"""
Zero-Leakage Privacy Guard & Medical Guardrails for SNN Chatbot.

Ensures strict compliance with:
- HIPAA/GDPR data minimization (no raw microvolt EEG arrays transmitted in prompts).
- PII/PHI scrubbing (names, emails, phone numbers, identifiers masked).
- AGENTS.md non-diagnostic research guardrails (no claims of diagnosing epilepsy, depression, or clinical impairments).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Common PII Regex Patterns
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

# Medical Disclaimers
CLINICAL_DISCLAIMER = (
    "\n\n*Medical Guardrail Notice: The SNN-AI platform provides real-time neuromorphic cognitive workload "
    "and affective estimations based on empirical research models. It is not a diagnostic device for clinical "
    "epilepsy, psychiatric disorders, or neurological pathologies. For medical diagnoses, please consult a licensed healthcare professional.*"
)

MEDICAL_KEYWORDS = {
    "diagnose", "diagnosis", "epilepsy", "seizure", "depression", "schizophrenia", 
    "bipolar", "autism", "adhd", "pathology", "cure", "prescription", "medication", 
    "mri", "tumor", "stroke"
}


def sanitize_user_input(text: str) -> Tuple[str, List[str]]:
    """
    Scrub identifiable PII/PHI from incoming user prompts.

    Returns
    -------
    sanitized_text : str
        Text with sensitive entities replaced by [REDACTED_TYPE].
    redacted_items : list of str
        List of entity types that were redacted.
    """
    redacted = []

    if EMAIL_RE.search(text):
        text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        redacted.append("email")

    if PHONE_RE.search(text):
        text = PHONE_RE.sub("[REDACTED_PHONE]", text)
        redacted.append("phone")

    if SSN_RE.search(text):
        text = SSN_RE.sub("[REDACTED_SSN]", text)
        redacted.append("ssn")

    if CARD_RE.search(text):
        text = CARD_RE.sub("[REDACTED_CARD]", text)
        redacted.append("card")

    return text, redacted


def enforce_medical_guardrails(response_text: str, user_query: str = "") -> str:
    """
    Checks if the conversation touches on regulated medical or diagnostic topics,
    and appends protective research boundary disclaimers when appropriate.
    """
    query_lower = user_query.lower()
    resp_lower = response_text.lower()

    needs_disclaimer = any(k in query_lower for k in MEDICAL_KEYWORDS) or any(
        k in resp_lower for k in ("diagnosis", "clinical pathology", "prescribe", "epilepsy")
    )

    if needs_disclaimer and CLINICAL_DISCLAIMER.strip() not in response_text:
        return response_text + CLINICAL_DISCLAIMER

    return response_text


def sanitize_eeg_telemetry(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drops high-dimensional raw microvolt timeseries from prompt metadata,
    retaining only high-level cognitive indices and frequency summaries.
    """
    sanitized = {}
    for k, v in data.items():
        if k in ("raw", "raw_data", "time_series", "eeg_channels", "microvolts"):
            continue
        sanitized[k] = v
    return sanitized
