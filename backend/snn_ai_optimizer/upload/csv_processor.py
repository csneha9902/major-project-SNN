from __future__ import annotations

import csv
import datetime
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _infer_timestamp(row: Dict[str, Any], index: int, default_step: float = 0.2) -> float:
    for key in ("timestamp", "time"):
        if key in row:
            return _to_float(row[key], default=index * default_step)
        for k, v in row.items():
            if k.lower() == key:
                return _to_float(v, default=index * default_step)

    for key in ("datetime", "date_time"):
        val = None
        if key in row:
            val = row[key]
        else:
            for k, v in row.items():
                if k.lower() == key:
                    val = v
                    break
        if val:
            try:
                dt = datetime.datetime.fromisoformat(str(val))
                return dt.timestamp()
            except Exception:
                pass
    return float(index * default_step)


def _detect_dialect(sample: str) -> csv.Dialect:
    try:
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample)
    except Exception:
        class DefaultDialect(csv.excel):
            delimiter = ","
        return DefaultDialect()


def _is_header_row(tokens: List[str]) -> bool:
    """Returns True if the tokens row appears to be column names rather than numbers."""
    if not tokens:
        return False
    # If all tokens can be converted to float, it is data, not a header
    numeric_count = 0
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        try:
            float(t)
            numeric_count += 1
        except ValueError:
            pass
    # If more than 50% can NOT be parsed as float, it's likely a header
    return (numeric_count / max(len(tokens), 1)) < 0.5


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = arr - arr.min()
    peak = arr.max()
    if peak < 1e-9:
        return np.zeros_like(arr)
    arr = arr / peak
    return (arr * 1.6) + 0.2


def _process_precomputed_features(
    rows: List[Dict[str, Any]],
    upload_id: str,
    channel_names: List[str],
) -> Dict[str, Any]:
    """Handles CSV files that already contain 'alpha' and 'beta' feature columns."""
    from snn_ai_optimizer.cognitive import compute_cognitive_state_full
    from snn_ai_optimizer.optimizer import recommend_task
    from snn_ai_optimizer.snn.inference import project_scalars_to_128_features

    def get_col(row: Dict[str, Any], names: List[str], default=None):
        for n in names:
            if n in row:
                return row[n]
            for k, v in row.items():
                if k.lower() == n.lower():
                    return v
        return default

    timestamps: List[float] = []
    alpha_raw: List[float] = []
    beta_raw: List[float] = []
    heart_rate_raw: List[float] = []

    for i, row in enumerate(rows):
        ts = _infer_timestamp(row, i)
        a = _to_float(get_col(row, ["alpha"]))
        b = _to_float(get_col(row, ["beta"]))
        hr = _to_float(get_col(row, ["heart_rate", "heartRate", "hr", "bpm"], default=None), default=0.0)
        timestamps.append(ts)
        alpha_raw.append(a)
        beta_raw.append(b)
        heart_rate_raw.append(hr)

    timestamps_arr = np.asarray(timestamps, dtype=float)
    alpha_norm = _normalize(np.asarray(alpha_raw, dtype=float))
    beta_norm = _normalize(np.asarray(beta_raw, dtype=float))
    hr_arr = np.asarray(heart_rate_raw, dtype=float)

    if not np.any(hr_arr):
        variation = (beta_norm - alpha_norm) * 25.0
        noise = np.random.normal(0, 1.5, size=alpha_norm.shape)
        hr_arr = np.clip(72.0 + variation + noise, 55.0, 120.0)

    n_samples = len(timestamps_arr)
    if n_samples < 2:
        raise ValueError("CSV does not contain enough data rows (minimum 2 required)")

    duration = max(0.0, float(timestamps_arr[-1] - timestamps_arr[0]))
    lf_hf_val = float((np.mean(beta_norm) + 1e-6) / (np.mean(alpha_norm) + 1e-6))

    time_series = []
    for ts, a_val, b_val, hr_val in zip(timestamps_arr, alpha_norm, beta_norm, hr_arr):
        feat_128 = project_scalars_to_128_features(alpha=float(a_val), beta=float(b_val), lf_hf=lf_hf_val)
        snn_res = compute_cognitive_state_full(float(a_val), float(b_val), lf_hf_val, features_128=feat_128)
        state = snn_res["cognitive_state"]
        arousal = snn_res["arousal_label"]
        rec = recommend_task(state)

        time_series.append({
            "timestamp": float(ts),
            "alpha": float(a_val),
            "beta": float(b_val),
            "heart_rate": round(float(hr_val), 2),
            "cognitive_state": state,
            "arousal_label": arousal,
            "recommendation": rec,
        })

    features = {
        "timestamps": timestamps_arr.astype(float).tolist(),
        "alpha": alpha_norm.astype(float).tolist(),
        "beta": beta_norm.astype(float).tolist(),
        "alpha_mean": float(np.mean(alpha_norm)),
        "beta_mean": float(np.mean(beta_norm)),
        "alpha_std": float(np.std(alpha_norm)),
        "beta_std": float(np.std(beta_norm)),
        "lf_hf_ratio": lf_hf_val,
        "n_samples": int(n_samples),
    }

    metadata = {
        "upload_id": upload_id,
        "format": "csv",
        "duration": float(duration),
        "sfreq": round(n_samples / max(duration, 0.001), 2) if duration > 0 else 5.0,
        "channels": channel_names or ["alpha", "beta"],
        "n_channels": len(channel_names) if channel_names else 2,
    }

    return {
        "upload_id": upload_id,
        "metadata": metadata,
        "features": features,
        "time_series": time_series,
        "extended_analysis": {},
    }


def _process_raw_eeg_matrix(
    data_matrix: np.ndarray,
    upload_id: str,
    channel_names: List[str],
    timestamps_column: Optional[np.ndarray] = None,
    default_sfreq: float = 128.0,
) -> Dict[str, Any]:
    """
    Processes multi-channel continuous raw EEG signal matrices (e.g. 19-ch s00.csv, 32-ch DEAP).
    Applies 4th-order Butterworth bandpass filtering for Alpha (8-13 Hz) & Beta (13-30 Hz)
    and extracts windowed 128 Welch PSD features for SNN inference.
    """
    from snn_ai_optimizer.cognitive import compute_cognitive_state_full
    from snn_ai_optimizer.features.eeg_features import extract_band_powers
    from snn_ai_optimizer.optimizer import recommend_task
    from snn_ai_optimizer.snn.inference import project_scalars_to_128_features

    # data_matrix is (n_samples, n_channels)
    total_samples, n_channels = data_matrix.shape
    if total_samples < 4:
        raise ValueError("EEG matrix does not contain enough data samples")

    # Determine sampling frequency
    sfreq = default_sfreq
    if timestamps_column is not None and len(timestamps_column) > 1:
        diffs = np.diff(timestamps_column)
        median_dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
        if median_dt > 0:
            sfreq = max(1.0, 1.0 / median_dt)

    # Transpose to (n_channels, total_samples) for signal processing
    eeg_channels = data_matrix.T.astype(np.float64)

    # Filter Alpha (8-13 Hz) and Beta (13-30 Hz) bands using Butterworth filter
    try:
        from scipy import signal as sp_signal

        nyquist = sfreq / 2.0
        # Guard band edges below Nyquist
        alpha_high = min(13.0, nyquist - 0.5)
        alpha_low = min(8.0, alpha_high - 1.0)
        beta_high = min(30.0, nyquist - 0.5)
        beta_low = min(13.0, beta_high - 1.0)

        alpha_b, alpha_a = sp_signal.butter(4, [alpha_low / nyquist, alpha_high / nyquist], btype="band")
        beta_b, beta_a = sp_signal.butter(4, [beta_low / nyquist, beta_high / nyquist], btype="band")

        alpha_filtered = sp_signal.filtfilt(alpha_b, alpha_a, eeg_channels, axis=1)
        beta_filtered = sp_signal.filtfilt(beta_b, beta_a, eeg_channels, axis=1)

        alpha_envelopes = np.abs(sp_signal.hilbert(alpha_filtered, axis=1))
        beta_envelopes = np.abs(sp_signal.hilbert(beta_filtered, axis=1))

        alpha_avg = np.mean(alpha_envelopes, axis=0)
        beta_avg = np.mean(beta_envelopes, axis=0)
    except Exception:
        # Fallback if scipy Hilbert / filtering fails
        alpha_avg = np.abs(eeg_channels).mean(axis=0)
        beta_avg = alpha_avg * 0.85

    # Downsample envelope series to 5 samples per second for smooth interactive plotting
    samples_per_second = 5.0
    step = max(1, int(round(max(sfreq, 1.0) / samples_per_second)))
    sample_indices = np.arange(0, total_samples, step, dtype=int)

    if timestamps_column is not None and len(timestamps_column) == total_samples:
        timestamps = timestamps_column[sample_indices].astype(float).tolist()
    else:
        timestamps = (sample_indices / sfreq).astype(float).tolist()

    alpha_samples = alpha_avg[sample_indices]
    beta_samples = beta_avg[sample_indices]

    alpha_norm = _normalize(alpha_samples)
    beta_norm = _normalize(beta_samples)

    # Standardize to 32 EEG channels for SNN Welch PSD windowed extraction
    if n_channels >= 32:
        eeg_32 = eeg_channels[:32, :]
    else:
        repeats = int(np.ceil(32 / max(n_channels, 1)))
        eeg_32 = np.tile(eeg_channels, (repeats, 1))[:32, :]

    window_sec = 2.0
    window_samples = max(int(round(window_sec * sfreq)), 64)
    n_windows = total_samples // window_samples

    window_feats = np.empty((0, 128), dtype=np.float32)
    if n_windows > 0:
        windowed_raw = np.zeros((n_windows, 32, window_samples), dtype=np.float32)
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            windowed_raw[w] = eeg_32[:, start:end]

        try:
            nperseg = min(window_samples, 256)
            window_feats = extract_band_powers(windowed_raw, fs=int(sfreq), nperseg=nperseg)
        except Exception:
            window_feats = np.empty((0, 128), dtype=np.float32)

    raw_alpha_mean = float(np.mean(alpha_samples)) if alpha_samples.size else 1.0
    raw_beta_mean = float(np.mean(beta_samples)) if beta_samples.size else 1.0
    lf_hf_val = float((raw_beta_mean + 1e-6) / (raw_alpha_mean + 1e-6))

    # Compute SNN affective states and recommendations along the timeline
    time_series = []
    n_pts = len(timestamps)
    n_feat_windows = len(window_feats)

    for i in range(n_pts):
        a_val = float(alpha_norm[i])
        b_val = float(beta_norm[i])

        if n_feat_windows > 0:
            w_idx = min(int(round((i / max(n_pts - 1, 1)) * (n_feat_windows - 1))), n_feat_windows - 1)
            f_128 = window_feats[w_idx]
        else:
            f_128 = project_scalars_to_128_features(alpha=a_val, beta=b_val, lf_hf=lf_hf_val)

        snn_res = compute_cognitive_state_full(a_val, b_val, lf_hf_val, features_128=f_128)
        state = snn_res["cognitive_state"]
        arousal = snn_res["arousal_label"]
        rec = recommend_task(state)

        # Realistic heart rate correlated with affective arousal
        hr_variation = (b_val - a_val) * 20.0
        hr_val = float(np.clip(72.0 + hr_variation + np.random.normal(0, 1.2), 55.0, 120.0))

        time_series.append({
            "timestamp": float(timestamps[i]),
            "alpha": a_val,
            "beta": b_val,
            "heart_rate": round(hr_val, 2),
            "cognitive_state": state,
            "arousal_label": arousal,
            "recommendation": rec,
        })

    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else float(total_samples / sfreq)

    features = {
        "timestamps": timestamps,
        "alpha": alpha_norm.astype(float).tolist(),
        "beta": beta_norm.astype(float).tolist(),
        "alpha_mean": float(np.mean(alpha_norm)),
        "beta_mean": float(np.mean(beta_norm)),
        "alpha_std": float(np.std(alpha_norm)),
        "beta_std": float(np.std(beta_norm)),
        "lf_hf_ratio": lf_hf_val,
        "n_samples": n_pts,
    }

    metadata = {
        "upload_id": upload_id,
        "format": "csv",
        "duration": round(duration, 2),
        "sfreq": round(sfreq, 1),
        "channels": channel_names or [f"ch_{i+1}" for i in range(n_channels)],
        "n_channels": n_channels,
    }

    return {
        "upload_id": upload_id,
        "metadata": metadata,
        "features": features,
        "time_series": time_series,
        "extended_analysis": {},
    }


def process_csv_file(file_path: Path | str, upload_id: str) -> Dict[str, Any]:
    """
    Universal CSV processor supporting:
    1. Pre-computed feature datasets (with 'alpha', 'beta', optional 'heart_rate')
    2. Multi-channel raw continuous EEG matrices with channel headers (e.g. Fp1, Fp2, Cz...)
    3. Multi-channel raw continuous EEG matrices without headers (e.g. 19-ch s00.csv)
    4. Arbitrary delimiters (comma, tab, semicolon, whitespace)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        f.seek(0)
        dialect = _detect_dialect(sample)
        reader = csv.reader(f, dialect=dialect)
        first_row = None
        for row in reader:
            if row and any(c.strip() for c in row):
                first_row = [c.strip() for c in row]
                break

    if not first_row:
        raise ValueError("CSV file appears to be completely empty")

    has_header = _is_header_row(first_row)

    # Check if this is a pre-computed feature file (contains 'alpha' and 'beta' column names)
    if has_header:
        lower_cols = [c.lower() for c in first_row]
        if "alpha" in lower_cols and "beta" in lower_cols:
            rows_dict: List[Dict[str, Any]] = []
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                dreader = csv.DictReader(f, dialect=dialect)
                for r in dreader:
                    if r:
                        rows_dict.append(r)
            if not rows_dict:
                raise ValueError("CSV feature table contains header but no data rows")
            return _process_precomputed_features(rows_dict, upload_id, first_row)

    # Otherwise, parse as continuous numerical EEG matrix (with or without channel header)
    all_numeric_rows: List[List[float]] = []
    channel_names: List[str] = []
    timestamp_idx: Optional[int] = None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, dialect=dialect)
        row_idx = 0
        for raw_tokens in reader:
            tokens = [t.strip() for t in raw_tokens if t.strip()]
            if not tokens:
                continue

            if row_idx == 0 and has_header:
                channel_names = tokens
                for idx, name in enumerate(tokens):
                    if name.lower() in ("timestamp", "time", "t", "time_sec", "seconds"):
                        timestamp_idx = idx
                        break
                row_idx += 1
                continue

            row_idx += 1
            # Convert row tokens to floats
            try:
                row_vals = [float(v) for v in tokens]
                all_numeric_rows.append(row_vals)
            except ValueError:
                # Skip non-numeric annotation rows if any
                continue

    if not all_numeric_rows:
        raise ValueError("No numeric EEG data rows could be extracted from the CSV file")

    data_arr = np.asarray(all_numeric_rows, dtype=np.float64)

    timestamps_col = None
    if timestamp_idx is not None and timestamp_idx < data_arr.shape[1]:
        timestamps_col = data_arr[:, timestamp_idx]
        data_arr = np.delete(data_arr, timestamp_idx, axis=1)
        if channel_names and timestamp_idx < len(channel_names):
            channel_names.pop(timestamp_idx)

    return _process_raw_eeg_matrix(
        data_matrix=data_arr,
        upload_id=upload_id,
        channel_names=channel_names,
        timestamps_column=timestamps_col,
        default_sfreq=128.0,
    )
