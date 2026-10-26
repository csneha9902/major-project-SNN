"""
Unit Tests for Neuromorphic SNN Chatbot, Neuron Inspector, Appointment Scheduler,
and Zero-Leakage Privacy Guard.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from snn_ai_optimizer.app import app
from snn_ai_optimizer.chatbot.privacy_guard import (
    sanitize_user_input,
    sanitize_eeg_telemetry,
    enforce_medical_guardrails,
    CLINICAL_DISCLAIMER,
)
from snn_ai_optimizer.chatbot.neuron_inspector import SNNNeuronInspector
from snn_ai_optimizer.chatbot.appointment_service import AppointmentService
from snn_ai_optimizer.chatbot.chat_engine import SNNChatEngine


@pytest.fixture
def test_client():
    return TestClient(app)


# =========================================================================
# 1. Privacy Guard & Medical Guardrail Tests
# =========================================================================

def test_privacy_guard_scrubbing():
    """Verify that PII such as email, phone, SSN, and cards are redacted."""
    sample_text = (
        "My email is patient.doe@hospital.org and phone is 415-555-2671. "
        "My SSN is 123-45-6789 and card is 4111-2222-3333-4444."
    )
    sanitized, redacted = sanitize_user_input(sample_text)

    assert "[REDACTED_EMAIL]" in sanitized
    assert "patient.doe@hospital.org" not in sanitized
    assert "[REDACTED_PHONE]" in sanitized
    assert "415-555-2671" not in sanitized
    assert "[REDACTED_SSN]" in sanitized
    assert "123-45-6789" not in sanitized
    assert "[REDACTED_CARD]" in sanitized
    assert "4111-2222-3333-4444" not in sanitized
    assert set(redacted) == {"email", "phone", "ssn", "card"}


def test_privacy_guard_eeg_telemetry_minimization():
    """Verify high-dimensional raw microvolt timeseries are dropped."""
    payload = {
        "user_id": "sub-01",
        "cognitive_state": "Focused",
        "alpha": 0.45,
        "beta": 0.78,
        "raw_data": [[12.4, -5.2], [10.1, -3.3]],
        "time_series": [1.0, 2.0, 3.0],
        "microvolts": [55.2, 42.1],
    }
    sanitized = sanitize_eeg_telemetry(payload)
    assert "cognitive_state" in sanitized
    assert "alpha" in sanitized
    assert "beta" in sanitized
    assert "raw_data" not in sanitized
    assert "time_series" not in sanitized
    assert "microvolts" not in sanitized


def test_enforce_medical_guardrails_diagnostic_disclaimer():
    """Verify medical/diagnostic queries automatically append protective research disclaimers."""
    query = "Can the SNN diagnose whether I have clinical epilepsy or seizures?"
    resp = "The model classifies high vs low arousal from DEAP frequency power."
    guarded = enforce_medical_guardrails(resp, user_query=query)

    assert CLINICAL_DISCLAIMER.strip() in guarded
    assert "Medical Guardrail Notice" in guarded


# =========================================================================
# 2. SNN Neuron Inspector Tests
# =========================================================================

def test_neuron_inspector_hidden1_neuron_8():
    """Verify deep synaptic weight inspection for Neuron 8 in Hidden LIF 1."""
    inspector = SNNNeuronInspector()
    explanation = inspector.explain_neuron(layer="hidden1", neuron_index=7)

    assert "Hidden LIF" in explanation["layer"]
    assert explanation["neuron_index"] == 7
    assert "bias" in explanation
    assert "dominant_band" in explanation
    assert "dominant_region" in explanation
    assert "top_excitatory" in explanation
    assert len(explanation["top_excitatory"]) > 0
    assert "summary" in explanation
    # Summary should mention receptive field or band dominance
    assert "Neuron 8" in explanation["summary"]


def test_neuron_inspector_hidden2_and_output():
    """Verify layer 2 and output layer inspection."""
    inspector = SNNNeuronInspector()
    
    # Hidden 2
    res2 = inspector.explain_neuron(layer="hidden2", neuron_index=15)
    assert "Hidden LIF Layer 2" in res2["layer"]
    assert "top_excitatory_upstream" in res2
    assert "summary" in res2

    # Output
    res_out = inspector.explain_neuron(layer="output", neuron_index=1)
    assert "Output" in res_out["layer"]
    assert "class_name" in res_out


# =========================================================================
# 3. Appointment Service Tests
# =========================================================================

def test_appointment_service_lifecycle(tmp_path, monkeypatch):
    """Verify appointment creation, listing, and cancellation with JSON persistence."""
    fake_file = tmp_path / "appointments.json"
    monkeypatch.setattr("snn_ai_optimizer.chatbot.appointment_service._APPOINTMENTS_FILE", fake_file)

    service = AppointmentService()
    # List initial seeded
    initial = service.list_appointments()
    assert len(initial) >= 2

    # Create new appointment
    created = service.create_appointment(
        patient_name="Alex Rivera",
        clinician="Dr. Chen (Neurophysiology)",
        date_str="2026-11-05",
        time_str="11:00 AM",
        reason="Follow-up cognitive workload session",
    )
    assert created["id"].startswith("apt-")
    assert created["patient_name"] == "Alex Rivera"

    # Verify presence in list
    all_apts = service.list_appointments()
    assert any(a["id"] == created["id"] for a in all_apts)

    # Cancel
    cancelled = service.cancel_appointment(created["id"])
    assert cancelled["status"] == "cancelled"

    # Verify active filter
    active = service.list_appointments(status="scheduled")
    assert not any(a["id"] == created["id"] for a in active)


# =========================================================================
# 4. SNNChatEngine Intent Routing Tests
# =========================================================================

def test_chat_engine_neuron_inspection_query():
    """Verify natural language query about neuron 8 in hidden lif 1 is routed correctly."""
    engine = SNNChatEngine()
    result = engine.process_message("Please explain neuron 8 in hidden lif 1 for me")

    assert result["intent"] == "neuron_inspection"
    assert "Neuron 8" in result["response"]
    assert result["action"] is not None
    assert result["action"]["neuron_index"] == 7  # 1-based user input 8 -> 0-based 7


def test_chat_engine_research_metrics_query():
    """Verify benchmark queries return empirical AUROC and SVM comparison without hallucinating."""
    engine = SNNChatEngine()
    result = engine.process_message("What is the AUROC benchmark score compared to SVM?")

    assert result["intent"] == "research_metrics"
    assert "0.5532" in result["response"]
    assert "0.4834" in result["response"]
    assert "88.14%" in result["response"]


def test_chat_engine_math_query():
    """Verify queries asking for SNN math return Poisson and LIF equations."""
    engine = SNNChatEngine()
    result = engine.process_message("What are the LIF formulas and surrogate gradient used in training?")

    assert result["intent"] == "snn_math"
    assert "Leaky Integrate-and-Fire" in result["response"]
    assert "ArcTangent (ATan)" in result["response"]


def test_chat_engine_appointment_scheduling():
    """Verify natural language appointment booking triggers the appointment handler."""
    engine = SNNChatEngine()
    result = engine.process_message("Please schedule an appointment with Dr. Sharma for tomorrow at 2 PM")

    assert result["intent"] == "appointment"
    assert "Appointment Confirmed" in result["response"] or "Scheduled" in result["response"]


# =========================================================================
# 5. FastAPI Endpoints Integration Tests
# =========================================================================

def test_api_chat_message_endpoint(test_client):
    """Test POST /api/chat/message endpoint."""
    resp = test_client.post(
        "/api/chat/message",
        json={"message": "Explain neuron 0 in hidden lif 1", "context": "research"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "intent" in data
    assert data["intent"] == "neuron_inspection"


def test_api_chat_appointments_endpoint(test_client):
    """Test GET and POST /api/chat/appointments."""
    # List
    list_resp = test_client.get("/api/chat/appointments")
    assert list_resp.status_code == 200
    assert "appointments" in list_resp.json()

    # Create
    create_resp = test_client.post(
        "/api/chat/appointments",
        json={
            "patient_name": "Test Subject",
            "clinician": "Dr. Testing",
            "date": "2026-11-12",
            "time": "09:00 AM",
            "reason": "Test SNN consultation",
        },
    )
    assert create_resp.status_code == 200
    created_id = create_resp.json()["appointment"]["id"]

    # Delete / Cancel
    del_resp = test_client.delete(f"/api/chat/appointments/{created_id}")
    assert del_resp.status_code == 200


def test_api_chat_neuron_direct_inspection(test_client):
    """Test GET /api/chat/neuron/hidden1/8 direct inspection endpoint."""
    resp = test_client.get("/api/chat/neuron/hidden1/8")
    assert resp.status_code == 200
    data = resp.json()
    assert data["neuron_index"] == 8
    assert "Hidden LIF" in data["layer"]
    assert "summary" in data
