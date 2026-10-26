"""
Clinical EEG Appointment & Consultation Scheduler Service.

Manages booking, rescheduling, listing, and cancellation of clinical
neuro-monitoring follow-up appointments with persistent JSON storage.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_APPOINTMENTS_FILE = Path(__file__).resolve().parent.parent.parent.parent / "results" / "history" / "appointments.json"


def _ensure_storage() -> Path:
    _APPOINTMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _APPOINTMENTS_FILE.exists():
        initial_data = [
            {
                "id": "apt-seed-1",
                "patient_name": "Active Learner / Subject",
                "clinician": "Dr. Aditi Sharma (Neurophysiology)",
                "date": (date.today() + timedelta(days=2)).isoformat(),
                "time": "10:30 AM",
                "reason": "EEG Cognitive Workload & SNN Arousal Review",
                "status": "scheduled",
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "apt-seed-2",
                "patient_name": "Active Learner / Subject",
                "clinician": "Dr. Marcus Vance (Cognitive Ergonomics)",
                "date": (date.today() + timedelta(days=5)).isoformat(),
                "time": "02:00 PM",
                "reason": "Adaptive Task Balancing & Fatigue Management",
                "status": "scheduled",
                "created_at": datetime.now().isoformat(),
            }
        ]
        _APPOINTMENTS_FILE.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")
    return _APPOINTMENTS_FILE


class AppointmentService:
    """Provides appointment scheduling capabilities for clinical neuro-monitoring."""

    def __init__(self) -> None:
        self.file_path = _ensure_storage()

    def _load_data(self) -> List[Dict[str, Any]]:
        try:
            content = self.file_path.read_text(encoding="utf-8")
            return json.loads(content)
        except Exception:
            return []

    def _save_data(self, data: List[Dict[str, Any]]) -> None:
        self.file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_appointments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        records = self._load_data()
        if status:
            return [r for r in records if r.get("status") == status]
        return records

    def create_appointment(
        self,
        patient_name: str,
        clinician: str,
        date_str: str,
        time_str: str,
        reason: str = "General Neuro-Monitoring Consultation",
    ) -> Dict[str, Any]:
        records = self._load_data()
        new_apt = {
            "id": f"apt-{uuid.uuid4().hex[:8]}",
            "patient_name": patient_name or "Anonymous Patient",
            "clinician": clinician or "Attending Neurophysiologist",
            "date": date_str,
            "time": time_str,
            "reason": reason,
            "status": "scheduled",
            "created_at": datetime.now().isoformat(),
        }
        records.append(new_apt)
        self._save_data(records)
        return new_apt

    def cancel_appointment(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        records = self._load_data()
        for r in records:
            if r["id"] == appointment_id:
                r["status"] = "cancelled"
                self._save_data(records)
                return r
        return None

    def reschedule_appointment(
        self,
        appointment_id: str,
        new_date: str,
        new_time: str,
    ) -> Optional[Dict[str, Any]]:
        records = self._load_data()
        for r in records:
            if r["id"] == appointment_id:
                r["date"] = new_date
                r["time"] = new_time
                r["status"] = "rescheduled"
                self._save_data(records)
                return r
        return None
