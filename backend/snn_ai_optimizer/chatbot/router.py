"""
FastAPI Router for Dual-Domain Chatbot Assistant & Appointment Services.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from snn_ai_optimizer.chatbot.appointment_service import AppointmentService
from snn_ai_optimizer.chatbot.chat_engine import SNNChatEngine
from snn_ai_optimizer.chatbot.neuron_inspector import SNNNeuronInspector

router = APIRouter(prefix="/api/chat", tags=["chatbot"])

_chat_engine = SNNChatEngine()
_appointment_service = AppointmentService()
_neuron_inspector = SNNNeuronInspector()


class ChatMessageRequest(BaseModel):
    message: str
    context: Optional[str] = "research"  # 'research' | 'dashboard'
    client_state: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None


class AppointmentCreateRequest(BaseModel):
    patient_name: Optional[str] = "Anonymous Patient"
    clinician: Optional[str] = "Attending Neurophysiologist"
    date: str
    time: str
    reason: Optional[str] = "General Neuro-Monitoring Consultation"


@router.post("/message")
async def send_chat_message(req: ChatMessageRequest):
    """
    Main conversational endpoint: processes queries across research and clinical domains,
    scrubbing PII, resolving multi-turn context, and enforcing medical guardrails.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    result = _chat_engine.process_message(
        message=req.message,
        context=req.context or "research",
        client_state=req.client_state or {},
        history=req.history or [],
    )
    return result


@router.get("/appointments")
async def list_appointments(status: Optional[str] = None):
    """Returns active scheduled clinical appointments."""
    return {"appointments": _appointment_service.list_appointments(status=status)}


@router.post("/appointments")
async def create_appointment_direct(req: AppointmentCreateRequest):
    """Creates a new clinical consultation appointment directly."""
    created = _appointment_service.create_appointment(
        patient_name=req.patient_name or "Anonymous Patient",
        clinician=req.clinician or "Attending Neurophysiologist",
        date_str=req.date,
        time_str=req.time,
        reason=req.reason or "General Consultation",
    )
    return {"status": "success", "appointment": created}


@router.delete("/appointments/{appointment_id}")
async def cancel_appointment(appointment_id: str):
    """Cancels a scheduled appointment by ID."""
    res = _appointment_service.cancel_appointment(appointment_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Appointment {appointment_id} not found.")
    return {"status": "success", "cancelled": res}


@router.get("/neuron/{layer}/{neuron_index}")
async def inspect_neuron(layer: str, neuron_index: int):
    """Directly inspects synaptic weights and receptive fields of any SNN neuron."""
    return _neuron_inspector.explain_neuron(layer=layer, neuron_index=neuron_index)
