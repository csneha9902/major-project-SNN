"""
Core Chatbot Brain & Conversational Engine.

Dispatches incoming natural language queries to:
1. Context-Aware Semantic Reasoning Engine (disaggregates graphs, point voltages,
   spike occurrences, post-spike dynamics, synaptic weights, and hypothetical dynamics).
2. Ground-Truth Data Access Layer (live continuous simulation traces, real weights,
   empirical benchmark records, and visualizer plotting metadata).
3. Clinical Appointment Scheduler (booking, listing, cancelling).
4. Answer Verification Engine & Privacy Guard (validating numerical claims, masking PII,
   and enforcing non-diagnostic medical boundaries).
"""

from __future__ import annotations

import re
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from snn_ai_optimizer.chatbot.appointment_service import AppointmentService
from snn_ai_optimizer.chatbot.conversation_context import ConversationContextManager, DialogueContext
from snn_ai_optimizer.chatbot.data_access import DataAccessLayer
from snn_ai_optimizer.chatbot.neuron_inspector import SNNNeuronInspector
from snn_ai_optimizer.chatbot.privacy_guard import sanitize_user_input
from snn_ai_optimizer.chatbot.reasoning_engine import SNNReasoningEngine
from snn_ai_optimizer.chatbot.verifier import AnswerVerifier
from snn_ai_optimizer.snn.inference import get_snn_inference_engine


class SNNChatEngine:
    """Intelligent, Context-Aware Assistant for SNN Research and Clinical Neuro-Monitoring."""

    def __init__(self) -> None:
        self.data_access = DataAccessLayer()
        self.inspector = self.data_access.inspector
        self.appointments = self.data_access.appointments
        self.engine = get_snn_inference_engine()
        self.reasoning = SNNReasoningEngine(self.data_access)
        self.verifier = AnswerVerifier()

    def process_message(
        self,
        message: str,
        context: str = "research",  # "research" or "dashboard"
        client_state: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message with multi-turn context resolution, live data binding,
        factual verification, and strict privacy protection.
        """
        # Step 1: Privacy scrub (mask email, phone, card, SSN)
        sanitized_msg, redacted_types = sanitize_user_input(message)
        msg_lower = sanitized_msg.lower()

        # Step 2: Appointment Intent (Scheduling, Listing, Cancelling)
        if any(k in msg_lower for k in ("schedule", "book", "appointment", "doctor", "consultation", "reschedule", "upcoming", "cancel appointment")):
            resp_text, action_data = self._handle_appointment_intent(sanitized_msg, msg_lower)
            result = {
                "intent": "appointment",
                "response": resp_text,
                "action": action_data,
            }
        else:
            # Step 3: Resolve multi-turn conversational context
            dialogue_context = ConversationContextManager.resolve_context(
                current_message=sanitized_msg,
                history=history,
                client_state=client_state,
            )

            # Step 4: Execute semantic reasoning engine
            result = self.reasoning.process_query(
                query=sanitized_msg,
                context=dialogue_context,
                client_state=client_state,
            )

        # Step 5: Answer verification & medical guardrails
        verified_result = self.verifier.verify_and_guard(
            response_data=result,
            raw_query=message,
        )

        guarded_response = verified_result.get("response", "")

        # Step 6: Prepend privacy redaction alert if PII was scrubbed
        if redacted_types:
            guarded_response = (
                f"> *[Privacy Guard Notice: Redacted sensitive {', '.join(redacted_types)} to maintain zero data leakage.]*\n\n"
                + guarded_response
            )

        return {
            "intent": verified_result.get("intent", "general_inquiry"),
            "response": guarded_response,
            "action": verified_result.get("action"),
            "sanitized_query": sanitized_msg,
            "redacted_types": redacted_types,
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_appointment_intent(self, raw_msg: str, msg_lower: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Parse appointment scheduling, listing, and cancellation intents."""
        # 1. Listing appointments
        if any(k in msg_lower for k in ("show", "list", "upcoming", "view", "check")) and "appointment" in msg_lower:
            apts = self.appointments.list_appointments(status="scheduled")
            if not apts:
                return "You currently have no scheduled appointments. Would you like to book an appointment with a neurophysiologist?", None

            lines = ["### Upcoming Clinical Consultations\n"]
            for a in apts:
                lines.append(f"* **{a['date']} at {a['time']}** — {a['clinician']}\n  *Reason: {a['reason']} (ID: `{a['id']}`)*")
            return "\n".join(lines), {"type": "list_appointments", "count": len(apts)}

        # 2. Cancelling an appointment
        cancel_match = re.search(r"cancel\s*(?:appointment)?\s*(apt-[a-z0-9]+)", msg_lower)
        if cancel_match:
            apt_id = cancel_match.group(1)
            res = self.appointments.cancel_appointment(apt_id)
            if res:
                return f"Appointment `{apt_id}` with {res['clinician']} on {res['date']} has been cancelled.", {"type": "cancel_appointment", "id": apt_id}
            return f"Could not find appointment `{apt_id}` to cancel.", None

        # 3. Booking a new appointment
        # Extract target date
        target_date = date.today() + timedelta(days=2)
        if "tomorrow" in msg_lower:
            target_date = date.today() + timedelta(days=1)
        elif "friday" in msg_lower:
            days_ahead = (4 - date.today().weekday()) % 7 or 7
            target_date = date.today() + timedelta(days=days_ahead)
        elif "monday" in msg_lower:
            days_ahead = (0 - date.today().weekday()) % 7 or 7
            target_date = date.today() + timedelta(days=days_ahead)
        elif "tuesday" in msg_lower:
            days_ahead = (1 - date.today().weekday()) % 7 or 7
            target_date = date.today() + timedelta(days=days_ahead)
        elif "wednesday" in msg_lower:
            days_ahead = (2 - date.today().weekday()) % 7 or 7
            target_date = date.today() + timedelta(days=days_ahead)
        elif "thursday" in msg_lower:
            days_ahead = (3 - date.today().weekday()) % 7 or 7
            target_date = date.today() + timedelta(days=days_ahead)

        # Extract time
        time_match = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm))", msg_lower)
        target_time = time_match.group(1).upper() if time_match else "10:00 AM"

        # Clinician
        clinician = "Dr. Aditi Sharma (Neurophysiologist)"
        if "vance" in msg_lower:
            clinician = "Dr. Marcus Vance (Cognitive Ergonomics)"

        reason = "EEG Cognitive Load & SNN Arousal Review"
        if "fatigue" in msg_lower:
            reason = "Persistent Cognitive Fatigue Consultation"
        elif "stress" in msg_lower:
            reason = "Arousal Spikes & Cognitive Strain Consultation"

        new_apt = self.appointments.create_appointment(
            patient_name="Active Learner / Subject",
            clinician=clinician,
            date_str=target_date.isoformat(),
            time_str=target_time,
            reason=reason,
        )

        resp = (
            f"### Appointment Confirmed!\n\n"
            f"Your clinical neuro-monitoring consultation has been successfully scheduled:\n"
            f"* **Doctor / Clinician:** {new_apt['clinician']}\n"
            f"* **Date:** {new_apt['date']}\n"
            f"* **Time:** {new_apt['time']}\n"
            f"* **Reason:** {new_apt['reason']}\n"
            f"* **Confirmation ID:** `{new_apt['id']}`\n\n"
            f"This has been saved to your clinical record. You can view or reschedule it anytime."
        )
        return resp, {"type": "appointment_created", "appointment": new_apt}
