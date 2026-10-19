from __future__ import annotations

import os
import time
import random
import math
from typing import Dict, Generator, Optional

import requests
import torch
import numpy as np
# Removed legacy SNN imports


from .cognitive import compute_cognitive_state, compute_cognitive_state_full
from .snn.inference import project_scalars_to_128_features
from .optimizer import recommend_task, _load_recommender


class DataStreamer:
    """Produces frames following the agreed API contract.

    Frame example:
    {
      "timestamp": 1678886400,
      "eeg": { "alpha": 0.6, "beta": 0.3 },
      "hrv": { "lf_hf_ratio": 0.8 },
      "cognitive_state": "Focused",
      "recommendation": { "task": "Review Chapter 3", "difficulty": 3 }
    }
    """

    def __init__(self):
        self.mode: str = "Stressed"  # Default to Stressed for exam crunch demo ("Focused" | "Stressed" | "Neutral")
        self.running: bool = True  # Controls if simulation is active
        self._latest: Optional[Dict] = None
        self._rng = random.Random(42)
        self.external_url = os.environ.get("EEG_SOURCE_URL")
        self.external_timeout = float(os.environ.get("EEG_SOURCE_TIMEOUT", "2.5"))
        self._http_session = requests.Session() if self.external_url else None
        self._source_status = {
            "external_configured": bool(self.external_url),
            "external_healthy": False,
            "using_external": False,
            "last_error": None,
            "external_url": self.external_url,
        }
        self._tracker = None
        self._tracker = None
                
        # Pre-warm the SNN recommender
        try:
            print("Pre-warming SNN recommender...")
            _load_recommender().predict("Neutral")
        except Exception as e:
            print(f"Failed to pre-warm SNN recommender: {e}")

    def set_tracker(self, tracker) -> None:
        self._tracker = tracker

    def set_mode(self, mode: str) -> None:
        if mode not in ("Focused", "Stressed", "Neutral"):
            mode = "Neutral"
        self.mode = mode
    
    def start_simulation(self) -> None:
        self.running = True
    
    def stop_simulation(self) -> None:
        self.running = False

    def get_latest(self) -> Optional[Dict]:
        return self._latest

    def get_ingestion_status(self) -> Dict:
        status = dict(self._source_status)
        status.setdefault("active_source", "external" if status.get("using_external") else "internal")
        return status

    def _build_dynamic_recommendation(self, state: str, alpha: float, beta: float, lf_hf: float, heart_rate: float) -> Dict:
        base_rec = recommend_task(state)
        
        if state == "Stressed":
            situation = (
                f"High Beta wave elevation ({beta:.2f}) with suppressed Alpha waves ({alpha:.2f}) and an elevated heart rate ({heart_rate:.0f} BPM, LF/HF {lf_hf:.2f}). "
                f"Your neural signals indicate acute cognitive stress typical during intensive exam prep crunching."
            )
            reasoning = "Excessive cognitive strain reduces working memory capacity and accelerates burnout. Lowering task difficulty and initiating brief relaxation intervals protects cognitive health."
            next_steps = [
                "Execute 3 minutes of 4-7-8 deep breathing to re-engage parasympathetic neural recovery.",
                "Switch to Tier 1/2 practice problems to consolidate retention without cognitive overload.",
                "Hydrate and step back from high-intensity problem solving for a 5-minute break."
            ]
            difficulty_tag = "Tier 1 - Reduced Load"
        elif state == "Focused":
            situation = (
                f"High Alpha-to-Beta synchronization ({alpha:.2f} α / {beta:.2f} β) with steady heart rate ({heart_rate:.0f} BPM). "
                f"Your brain is currently operating in an optimal cognitive flow state."
            )
            reasoning = "Peak cognitive performance window is active. Ideal timing for tackling complex problem sets or learning difficult new exam concepts."
            next_steps = [
                "Attempt higher-tier problem sets or complex exam chapters while focus is peak.",
                "Maintain an uninterrupted 25-minute Pomodoro study block.",
                "Minimize environmental distractions to preserve deep focus engagement."
            ]
            difficulty_tag = "Tier 4 - Deep Problem Solving"
        else:  # Neutral
            situation = (
                f"Balanced baseline EEG signal distribution (Alpha {alpha:.2f}, Beta {beta:.2f}, Heart Rate {heart_rate:.0f} BPM). "
                f"Cognitive state is calm, stable, and ready for structured study."
            )
            reasoning = "Neural activity is steady. Moderate difficulty practice maintains steady learning velocity without causing strain."
            next_steps = [
                "Review core concept summaries before advancing to timed practice sets.",
                "Maintain steady 15-minute study intervals with brief check-ins.",
                "Ensure ergonomic posture to maintain optimal cerebral oxygenation."
            ]
            difficulty_tag = "Tier 2 - Moderate Steady Load"

        return {
            **base_rec,
            "situation": situation,
            "reasoning": reasoning,
            "next_steps": next_steps,
            "difficulty_tag": difficulty_tag,
            "state": state
        }

    def _sample_alpha_beta(self) -> (float, float):
        # Create smoother, more stable patterns with controlled variation
        t = time.time()
        # Use slower, smoother frequencies for more stable patterns
        freq1 = 0.2  # Slower base frequency
        freq2 = 0.5  # Medium frequency
        freq3 = 1.0  # Faster but still smooth
        
        # Smaller, more controlled amplitudes
        amp1 = 0.15
        amp2 = 0.10
        amp3 = 0.08
        
        # Smooth sinusoidal drifts
        drift1 = math.sin(t * freq1) * amp1
        drift2 = math.sin(t * freq2 + math.pi / 3) * amp2
        drift3 = math.sin(t * freq3 + math.pi / 6) * amp3
        combined_drift = drift1 + drift2 + drift3
        
        # Reduced random noise for smoother lines
        random_noise = self._rng.gauss(0, 0.08)  # Reduced from 0.2 to 0.08
        
        alpha_base = 0.5 + random_noise + combined_drift
        # Beta has inverse relationship with smoother variation
        beta_drift = -combined_drift * 0.6
        beta_base = 0.5 + self._rng.gauss(0, 0.08) + beta_drift  # Reduced noise
        
        if self.mode == "Focused":
            alpha_base += 0.3 + self._rng.uniform(-0.05, 0.05)
            beta_base -= 0.1 + self._rng.uniform(-0.02, 0.02)
        elif self.mode == "Stressed":
            alpha_base -= 0.1 + self._rng.uniform(-0.02, 0.02)
            beta_base += 0.3 + self._rng.uniform(-0.05, 0.05)
        
        # clamp with wider range
        alpha = max(0.1, min(1.8, alpha_base))
        beta = max(0.1, min(1.8, beta_base))
        return alpha, beta

    def _sample_lf_hf(self) -> float:
        # Smoother HRV patterns with controlled variation
        t = time.time()
        # Use slower, more stable frequencies
        freq1 = 0.25  # Slower base frequency
        freq2 = 0.6   # Medium frequency
        freq3 = 1.2   # Faster but still smooth
        
        # Smaller, more controlled amplitudes
        amp1 = 0.3
        amp2 = 0.2
        amp3 = 0.15
        
        # Smooth sinusoidal drifts
        drift1 = math.sin(t * freq1) * amp1
        drift2 = math.sin(t * freq2 + math.pi / 4) * amp2
        drift3 = math.sin(t * freq3 + math.pi / 2) * amp3
        combined_drift = drift1 + drift2 + drift3
        
        # Reduced random noise for smoother lines
        random_noise = self._rng.gauss(0, 0.12)  # Reduced from 0.3 to 0.12
        
        base = 1.0 + random_noise + combined_drift
        if self.mode == "Stressed":
            base += 0.8 + self._rng.uniform(-0.05, 0.05)  # Reduced variation
        elif self.mode == "Focused":
            base -= 0.25 + self._rng.uniform(-0.03, 0.03)  # Reduced variation
        return max(0.1, min(3.5, base))

    def _fetch_external_eeg(self):
        if not self._http_session or not self.external_url:
            return None

        url = self.external_url
        params = None
        if "{mode}" in url:
            url = url.format(mode=self.mode)
        else:
            params = {"mode": self.mode}

        try:
            response = self._http_session.get(url, params=params, timeout=self.external_timeout)
            response.raise_for_status()
            payload = response.json()

            container = payload.get("eeg") if isinstance(payload, dict) else None
            alpha = container.get("alpha") if container else payload.get("alpha")
            beta = container.get("beta") if container else payload.get("beta")
            hrv = payload.get("hrv") or {}
            lf_hf = hrv.get("lf_hf_ratio")

            if alpha is None or beta is None:
                raise ValueError("missing alpha/beta in payload")

            alpha = float(alpha)
            beta = float(beta)
            lf_hf = float(lf_hf) if lf_hf is not None else None

            self._source_status.update({
                "external_healthy": True,
                "using_external": True,
                "last_error": None,
                "last_success_ts": int(time.time()),
            })
            return alpha, beta, lf_hf
        except Exception as exc:
            self._source_status.update({
                "external_healthy": False,
                "using_external": False,
                "last_error": str(exc),
            })
            return None

    def stream(self, interval_sec: float = 0.5) -> Generator[Dict, None, None]:
        # Simple infinite generator
        while True:
            if self.running:
                ts = int(time.time())

                # Default to internally generated samples
                alpha, beta = self._sample_alpha_beta()
                lf_hf = self._sample_lf_hf()

                external_values = self._fetch_external_eeg()
                if external_values is not None:
                    alpha, beta, lf_hf_external = external_values
                    if lf_hf_external is not None:
                        lf_hf = lf_hf_external

                if external_values is None and not self._source_status.get("external_configured"):
                    self._source_status.update({
                        "using_external": False,
                        "external_healthy": False,
                    })

                # 128-dimensional band-power feature representation for SNN inference
                features_128 = project_scalars_to_128_features(alpha=alpha, beta=beta, lf_hf=lf_hf)
                snn_res = compute_cognitive_state_full(alpha, beta, lf_hf, features_128=features_128)
                state = snn_res["cognitive_state"]
                    
                # Derive heart rate BPM with smoother, more stable variation
                t = time.time()
                # Use slower, smoother frequencies for more stable heart rate
                hr_freq1 = 0.1   # Very slow base variation
                hr_freq2 = 0.4   # Medium frequency
                hr_freq3 = 0.9   # Faster but still smooth
                
                # Smaller, more controlled amplitudes
                hr_amp1 = 4
                hr_amp2 = 2.5
                hr_amp3 = 1.5
                
                # Smooth sinusoidal drifts
                hr_drift1 = math.sin(t * hr_freq1) * hr_amp1
                hr_drift2 = math.sin(t * hr_freq2 + math.pi / 3) * hr_amp2
                hr_drift3 = math.sin(t * hr_freq3 + math.pi / 6) * hr_amp3
                hr_combined = hr_drift1 + hr_drift2 + hr_drift3
                
                # Base heart rate varies with LF/HF ratio and mode
                base_hr = 70 + (lf_hf * 6)  # Reduced multiplier for less variation
                hr_variation = hr_combined + self._rng.gauss(0, 1.2)  # Reduced noise from 5 to 1.2
                
                heart_rate_bpm = max(55, min(115, base_hr + hr_variation))

                rec = self._build_dynamic_recommendation(state, alpha, beta, lf_hf, heart_rate_bpm)

                frame = {
                    "timestamp": ts,
                    "eeg": {"alpha": round(alpha, 3), "beta": round(beta, 3)},
                    "hrv": {"lf_hf_ratio": round(lf_hf, 3), "heart_rate_bpm": round(heart_rate_bpm, 2)},
                    "cognitive_state": state,
                    "recommendation": rec,
                    "ingestion": self.get_ingestion_status(),
                    "snn_inference": {
                        "arousal_label": snn_res["arousal_label"],
                        "confidence": snn_res["confidence"],
                        "using_snn": snn_res["using_snn"],
                    },
                }
                self._latest = frame
                if self._tracker:
                    try:
                        self._tracker.record_state(state, ts)
                    except Exception:
                        pass
                yield frame
            else:
                # When stopped, yield the last frame or null data
                if self._latest:
                    yield self._latest
                else:
                    yield {
                        "timestamp": int(time.time()),
                        "eeg": {"alpha": 0, "beta": 0},
                        "hrv": {"lf_hf_ratio": 0},
                        "cognitive_state": "Paused",
                        "recommendation": {"task": "Simulation Paused", "difficulty": 0},
                        "ingestion": self.get_ingestion_status(),
                    }
            time.sleep(interval_sec)
