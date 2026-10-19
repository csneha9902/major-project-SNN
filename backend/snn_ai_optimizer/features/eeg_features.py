"""
EEG band-power feature extraction using Welch's Power Spectral Density.

Design
------
Input : ndarray, shape (N, 32, 8064)
            N     = number of trials
            32    = EEG channels (first 32 of DEAP's 40-channel layout)
            8064  = 63 s × 128 Hz samples per trial

Output: ndarray, shape (N, 128)
            128 = 32 channels × 4 frequency bands
            Band order (within each channel block):
                0 → delta (1–4 Hz)
                1 → theta (4–8 Hz)
                2 → alpha (8–13 Hz)
                3 → beta  (13–30 Hz)

Feature vector layout
---------------------
Feature index k = channel_idx * 4 + band_idx

    ch0_delta, ch0_theta, ch0_alpha, ch0_beta,
    ch1_delta, ch1_theta, ch1_alpha, ch1_beta,
    ...
    ch31_delta, ch31_theta, ch31_alpha, ch31_beta

Preprocessing contract
-----------------------
This module performs NO normalisation.  A ``StandardScaler`` must be fitted
ONLY on training data and applied to both train and test features to prevent
data leakage.  See ``baseline.py`` for the correct usage pattern.

Research integrity
------------------
* No synthetic data is produced here.
* Identical preprocessing must be used at training and inference (AGENTS.md rule 10).
* Band limits match standard EEG neuroscience conventions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Canonical EEG frequency bands
# ---------------------------------------------------------------------------

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

# Ordered list for deterministic feature-vector construction
BAND_ORDER: Tuple[str, ...] = ("delta", "theta", "alpha", "beta")

# DEAP sampling rate (Hz)
SAMPLING_RATE: int = 128

# Number of EEG channels used
N_EEG_CHANNELS: int = 32

# Feature dimension: channels × bands
N_FEATURES: int = N_EEG_CHANNELS * len(BAND_ORDER)  # 128


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """
    Integrate PSD within [low, high) Hz using the trapezoidal rule.

    Parameters
    ----------
    freqs : ndarray, shape (F,)
        Frequency bins from ``scipy.signal.welch``.
    psd : ndarray, shape (F,)
        Power spectral density for a single channel.
    low, high : float
        Lower and upper frequency bounds (Hz).

    Returns
    -------
    float
        Band power (area under PSD curve in the band).
    """
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_band_powers(
    X: np.ndarray,
    fs: int = SAMPLING_RATE,
    nperseg: int = 256,
) -> np.ndarray:
    """
    Extract Welch-PSD band-power features from DEAP EEG trials.

    Parameters
    ----------
    X : ndarray, shape (N, 32, 8064)
        Raw EEG trials.  Values are in the original DEAP float32 units.
    fs : int, optional
        Sampling frequency in Hz.  Default ``128`` (DEAP standard).
    nperseg : int, optional
        Welch segment length.  Default ``256`` (2 s at 128 Hz).

    Returns
    -------
    features : ndarray, shape (N, 128)
        Band-power feature matrix.  Dtype is ``float64``.

    Raises
    ------
    ValueError
        If ``X`` does not have exactly 3 dimensions or the channel axis
        does not match ``N_EEG_CHANNELS``.

    Notes
    -----
    * Feature order: ch0_delta, ch0_theta, ch0_alpha, ch0_beta, ch1_delta, ...
    * No normalisation is applied.  Fit a ``StandardScaler`` on training
      data only, then transform both train and test sets.
    """
    if X.ndim != 3:
        raise ValueError(
            f"X must be 3-dimensional (N, channels, samples), got shape {X.shape}."
        )
    n_trials, n_channels, n_samples = X.shape
    if n_channels != N_EEG_CHANNELS:
        raise ValueError(
            f"Expected {N_EEG_CHANNELS} EEG channels, got {n_channels}. "
            "Ensure peripheral channels have been removed before feature extraction."
        )

    features = np.zeros((n_trials, N_FEATURES), dtype=np.float64)

    for trial_idx in range(n_trials):
        feat_idx = 0
        for ch in range(n_channels):
            signal = X[trial_idx, ch, :].astype(np.float64)
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
            for band_name in BAND_ORDER:
                low, high = BANDS[band_name]
                features[trial_idx, feat_idx] = _band_power(freqs, psd, low, high)
                feat_idx += 1

    return features


def feature_names() -> list[str]:
    """
    Return human-readable names for the 128 features in extraction order.

    Returns
    -------
    list of str
        e.g. ``['ch00_delta', 'ch00_theta', ..., 'ch31_beta']``
    """
    names = []
    for ch in range(N_EEG_CHANNELS):
        for band in BAND_ORDER:
            names.append(f"ch{ch:02d}_{band}")
    return names


def extract_windowed_band_powers(
    X: np.ndarray,
    y: np.ndarray | None = None,
    window_sec: float = 4.0,
    step_sec: float = 2.0,
    baseline_sec: float = 3.0,
    fs: int = SAMPLING_RATE,
    nperseg: int = 256,
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, int]:
    """
    Extract 128 Welch-PSD band-power features across temporal sliding windows.

    Transforms full 63-second DEAP trials into multiple overlapping windows:
    - Default 4.0-second window (512 samples at 128 Hz)
    - Default 2.0-second stride (256 samples, 50% overlap)
    - Pre-trial 3.0 seconds discarded (stimulus presentation: 60 s = 7,680 samples)
    - Yields exactly 29 windows per trial: (7,680 - 512) // 256 + 1 = 29.
    - Expands 480 trials into 13,920 feature vectors of canonical 128 dimensions.

    Parameters
    ----------
    X : ndarray, shape (N, 32, 8064)
        Raw EEG trials.
    y : ndarray or None, shape (N,)
        Optional trial-level labels to repeat for each temporal window.
    window_sec : float, default 4.0
        Duration of each temporal analysis window in seconds.
    step_sec : float, default 2.0
        Step / stride between successive window onsets in seconds.
    baseline_sec : float, default 3.0
        Pre-trial baseline duration to exclude in seconds.
    fs : int, default 128
        Sampling frequency in Hz.
    nperseg : int, default 256
        Welch segment length.

    Returns
    -------
    If y is provided:
        (features, y_windowed, n_windows_per_trial)
    If y is None:
        (features, n_windows_per_trial)
    """
    if X.ndim != 3:
        raise ValueError(
            f"X must be 3-dimensional (N, channels, samples), got shape {X.shape}."
        )
    n_trials, n_channels, n_samples = X.shape
    if n_channels != N_EEG_CHANNELS:
        raise ValueError(
            f"Expected {N_EEG_CHANNELS} EEG channels, got {n_channels}."
        )

    baseline_samples = int(baseline_sec * fs)
    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    if baseline_samples >= n_samples:
        raise ValueError(
            f"baseline_sec={baseline_sec} exceeds total sample duration."
        )

    active_data = X[:, :, baseline_samples:]
    stim_samples = active_data.shape[-1]

    if stim_samples < win_samples:
        raise ValueError(
            f"Stimulus duration ({stim_samples} samples) is shorter than window ({win_samples} samples)."
        )

    windows = []
    for start in range(0, stim_samples - win_samples + 1, step_samples):
        windows.append(active_data[:, :, start : start + win_samples])

    windows_arr = np.stack(windows, axis=1)  # shape (N, n_win, 32, win_samples)
    n_win_per_trial = windows_arr.shape[1]

    # Flatten to (N * n_win, 32, win_samples) for vectorized Welch
    flat_win = windows_arr.reshape(-1, n_channels, win_samples).astype(np.float64)

    freqs, psd = welch(flat_win, fs=fs, nperseg=min(nperseg, win_samples), axis=-1)

    band_list = []
    for band_name in BAND_ORDER:
        low, high = BANDS[band_name]
        mask = (freqs >= low) & (freqs < high)
        bp = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
        band_list.append(bp)

    features = np.stack(band_list, axis=2).reshape(-1, N_FEATURES)

    if y is not None:
        y_rep = np.repeat(y, n_win_per_trial)
        return features, y_rep, n_win_per_trial

    return features, n_win_per_trial

