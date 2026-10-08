from __future__ import annotations

try:
    import mne
    HAS_MNE = True
except ImportError:
    mne = None
    HAS_MNE = False

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import uuid
from datetime import datetime


class EDFProcessor:
    """Process EDF files and extract features for analysis."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.raw = None
        self.metadata = {}
        self.features = {}

    def load(self) -> None:
        """Load EDF file using MNE."""
        if not HAS_MNE:
            raise ValueError("MNE library is required to process EDF files. Please install mne: pip install mne")
        try:
            self.raw = mne.io.read_raw_edf(str(self.file_path), preload=True, verbose=False)
            self.metadata = {
                "n_channels": len(self.raw.ch_names),
                "sfreq": float(self.raw.info["sfreq"]),
                "duration": float(self.raw.times[-1]),
                "ch_names": self.raw.ch_names,
            }
        except Exception as e:
            raise ValueError(f"Failed to load EDF file: {str(e)}")


    def extract_features(self) -> Dict:
        """Extract alpha, beta, and other features from EDF."""
        if self.raw is None:
            raise ValueError("EDF file not loaded. Call load() first.")

        data = self.raw.get_data()
        sfreq = float(self.metadata["sfreq"])
        duration = float(self.metadata["duration"])

        def _normalize(values: np.ndarray) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return arr
            arr = arr - arr.min()
            peak = arr.max()
            if peak < 1e-9:
                return np.zeros_like(arr)
            arr = arr / peak  # 0-1 range
            return (arr * 1.6) + 0.2  # stretch to 0.2-1.8 for better visual contrast

        try:
            from scipy import signal as sp_signal

            nyquist = sfreq / 2.0
            alpha_b, alpha_a = sp_signal.butter(4, [8 / nyquist, 13 / nyquist], btype="band")
            beta_b, beta_a = sp_signal.butter(4, [13 / nyquist, 30 / nyquist], btype="band")

            alpha_envelopes = []
            beta_envelopes = []
            for ch_data in data:
                alpha_filtered = sp_signal.filtfilt(alpha_b, alpha_a, ch_data)
                beta_filtered = sp_signal.filtfilt(beta_b, beta_a, ch_data)
                alpha_envelopes.append(np.abs(sp_signal.hilbert(alpha_filtered)))
                beta_envelopes.append(np.abs(sp_signal.hilbert(beta_filtered)))

            alpha_avg = np.mean(alpha_envelopes, axis=0)
            beta_avg = np.mean(beta_envelopes, axis=0)
        except Exception:
            # Fall back to simple absolute amplitudes if scipy is unavailable
            alpha_avg = np.abs(data).mean(axis=0)
            beta_avg = alpha_avg

        samples_per_second = 5.0
        step = max(1, int(round(max(sfreq, 1.0) / samples_per_second)))

        if alpha_avg.size == 0:
            sample_indices = np.array([], dtype=int)
        else:
            sample_indices = np.arange(0, alpha_avg.shape[0], step, dtype=int)

        if sample_indices.size == 0:
            timestamps = []
            alpha_samples = np.array([])
            beta_samples = np.array([])
        else:
            timestamps = (sample_indices / sfreq).astype(float).tolist()
            alpha_samples = alpha_avg[sample_indices]
            beta_samples = beta_avg[sample_indices]

        alpha_series = _normalize(alpha_samples).tolist() if alpha_samples.size else []
        beta_series = _normalize(beta_samples).tolist() if beta_samples.size else []

        raw_alpha_mean = float(np.mean(alpha_samples)) if alpha_samples.size else 1.0
        raw_beta_mean = float(np.mean(beta_samples)) if beta_samples.size else 1.0
        lf_hf_ratio = float((raw_beta_mean + 1e-6) / (raw_alpha_mean + 1e-6))

        if timestamps:
            self.metadata["duration"] = float(timestamps[-1])
        else:
            self.metadata["duration"] = duration

        self.features = {
            "timestamps": timestamps,
            "alpha": alpha_series,
            "beta": beta_series,
            "alpha_mean": float(np.mean(alpha_series)) if alpha_series else 0.0,
            "beta_mean": float(np.mean(beta_series)) if beta_series else 0.0,
            "alpha_std": float(np.std(alpha_series)) if alpha_series else 0.0,
            "beta_std": float(np.std(beta_series)) if beta_series else 0.0,
            "lf_hf_ratio": lf_hf_ratio,
            "n_samples": len(timestamps),
        }

        return self.features

    def extract_windowed_128_features(self, window_sec: float = 2.0) -> np.ndarray:
        """
        Extract 128 Welch PSD band-power features across temporal windows.

        Returns
        -------
        ndarray of shape (N_windows, 128)
        """
        if self.raw is None:
            return np.empty((0, 128), dtype=np.float32)

        data = self.raw.get_data()
        sfreq = float(self.metadata.get("sfreq", 128.0))
        n_channels, total_samples = data.shape

        # Standardize to 32 EEG channels (repeat or truncate)
        if n_channels >= 32:
            data_32 = data[:32, :]
        else:
            repeats = int(np.ceil(32 / n_channels))
            data_32 = np.tile(data, (repeats, 1))[:32, :]

        window_samples = max(int(round(window_sec * sfreq)), 64)
        n_windows = total_samples // window_samples

        if n_windows == 0:
            return np.empty((0, 128), dtype=np.float32)

        windowed_raw = np.zeros((n_windows, 32, window_samples), dtype=np.float32)
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            windowed_raw[w] = data_32[:, start:end]

        from snn_ai_optimizer.features.eeg_features import extract_band_powers
        nperseg = min(window_samples, 256)
        return extract_band_powers(windowed_raw, fs=int(sfreq), nperseg=nperseg)

    def get_analysis_data(self) -> Dict:
        """Get formatted data ready for analysis."""
        if not self.features:
            self.extract_features()

        from snn_ai_optimizer.cognitive import compute_cognitive_state_full
        from snn_ai_optimizer.snn.inference import project_scalars_to_128_features
        from snn_ai_optimizer.optimizer import recommend_task

        window_feats = self.extract_windowed_128_features()
        n_windows = len(window_feats)

        states = []
        arousals = []
        recommendations = []
        heart_rates = []

        for i, (alpha, beta) in enumerate(zip(self.features["alpha"], self.features["beta"])):
            lf_hf = self.features["lf_hf_ratio"] + float(np.random.normal(0, 0.05))

            if n_windows > 0:
                # Map timestamp index proportionally to available windowed 128-features
                w_idx = min(int(round((i / max(len(self.features["alpha"]) - 1, 1)) * (n_windows - 1))), n_windows - 1)
                feat_128 = window_feats[w_idx]
            else:
                feat_128 = project_scalars_to_128_features(alpha=alpha, beta=beta, lf_hf=lf_hf)

            snn_res = compute_cognitive_state_full(alpha, beta, lf_hf, features_128=feat_128)
            state = snn_res["cognitive_state"]
            arousal = snn_res["arousal_label"]

            rec = recommend_task(state)
            hr = 60 + (lf_hf * 20) + float(np.random.normal(0, 2))

            states.append(state)
            arousals.append(arousal)
            recommendations.append(rec)
            heart_rates.append(round(float(hr), 2))

        return {
            "metadata": self.metadata,
            "features": self.features,
            "time_series": [
                {
                    "timestamp": ts,
                    "alpha": alpha,
                    "beta": beta,
                    "heart_rate": hr,
                    "cognitive_state": state,
                    "arousal_label": arousal,
                    "recommendation": rec,
                }
                for ts, alpha, beta, hr, state, arousal, rec in zip(
                    self.features["timestamps"],
                    self.features["alpha"],
                    self.features["beta"],
                    heart_rates,
                    states,
                    arousals,
                    recommendations,
                )
            ],
        }


def process_edf_file(file_path: str | Path, upload_id: str | None = None) -> Dict:
    """Process an EDF file and return analysis data."""
    processor = EDFProcessor(file_path)
    processor.load()
    analysis_data = processor.get_analysis_data()
    
    if upload_id:
        analysis_data["upload_id"] = upload_id
    else:
        analysis_data["upload_id"] = str(uuid.uuid4())
    
    analysis_data["processed_at"] = datetime.now().isoformat()
    return analysis_data

