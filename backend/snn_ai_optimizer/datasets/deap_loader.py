"""
DEAP dataset loader for the EEG research pipeline.

Loads preprocessed DEAP .dat files (Python pickle format) produced by the
DEAP authors' preprocessing script.  Each file corresponds to one participant
and contains:

    data   : ndarray, shape (40, 40, 8064)
               40 trials × 40 channels × 8064 time samples (63 s @ 128 Hz)
               Channels 0–31   → EEG (32 channels)
               Channels 32–39  → Peripheral physiological (8 channels)

    labels : ndarray, shape (40, 4)
               40 trials × 4 self-assessment ratings
               Column 0 → valence   (1–9)
               Column 1 → arousal   (1–9)
               Column 2 → dominance (1–9)
               Column 3 → liking    (1–9)

This module ONLY loads and validates raw data. It does NOT:
  - train any model
  - fit preprocessing transforms
  - normalise or standardise data
  - substitute synthetic/mock data on failure
  - download data from the internet

Usage
-----
    from snn_ai_optimizer.datasets.deap_loader import DEAPLoader

    loader = DEAPLoader()
    subject_ids = loader.discover_subjects()          # e.g. ['s01', 's02', ...]
    record = loader.load_subject('s01')
    # record['eeg']      shape (40, 32, 8064)
    # record['ratings']  shape (40, 4)
    # record['subject_id'] == 's01'
    # record['trial_ids']  == list(range(40))   (0-based: 0–39)
    # record['n_total_channels'] == 40
"""

import os
import pickle
import re
from typing import Dict, List, Any

import numpy as np

from snn_ai_optimizer.datasets.deap_config import (
    get_deap_data_dir,
    EXPECTED_TRIALS,
    EXPECTED_TOTAL_CHANNELS,
    EXPECTED_EEG_CHANNELS,
    EXPECTED_SAMPLES,
    EXPECTED_RATINGS,
)

# Regex that matches the DEAP subject filename convention: s01.dat … s32.dat
_SUBJECT_FILENAME_RE = re.compile(r"^(s\d{2})\.dat$", re.IGNORECASE)


class DEAPLoader:
    """
    Validates and loads DEAP preprocessed .dat files from ``DEAP_DATA_DIR``.

    Parameters
    ----------
    data_dir : str or None
        Path to the DEAP data directory.  When *None* (default), the value is
        read from the ``DEAP_DATA_DIR`` environment variable.

    Raises
    ------
    EnvironmentError
        If ``DEAP_DATA_DIR`` is not set and *data_dir* is not provided.
    FileNotFoundError
        If the directory does not exist on disk.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        if data_dir is not None:
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(
                    f"Provided data_dir='{data_dir}' does not exist or is not a "
                    "directory."
                )
            self._data_dir = data_dir
        else:
            self._data_dir = get_deap_data_dir()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def data_dir(self) -> str:
        """Absolute path to the DEAP data directory (read-only)."""
        return self._data_dir

    def discover_subjects(self) -> List[str]:
        """
        Return a sorted list of subject IDs discovered from the data directory.

        Subject IDs are derived from filenames matching the pattern ``sNN.dat``
        (e.g. ``s01.dat`` → ``'s01'``).  Files with other names are ignored.

        Returns
        -------
        list of str
            Sorted subject IDs, e.g. ``['s01', 's02', 's03', ...]``.

        Raises
        ------
        FileNotFoundError
            If no valid DEAP subject files are found in the directory.
        """
        entries = os.listdir(self._data_dir)
        subject_ids = []
        for entry in entries:
            m = _SUBJECT_FILENAME_RE.match(entry)
            if m:
                subject_ids.append(m.group(1).lower())
        subject_ids.sort()
        if not subject_ids:
            raise FileNotFoundError(
                f"No DEAP subject files (sNN.dat) found in '{self._data_dir}'. "
                "Ensure the DEAP preprocessed .dat files are present."
            )
        return subject_ids

    def load_subject(self, subject_id: str) -> Dict[str, Any]:
        """
        Load a single DEAP subject file and return a validated record.

        Subject ID is derived from the filename — not from array position —
        to prevent silent mis-identification.

        Trial IDs use a **zero-based** convention (0–39) matching Python array
        indices.  Trial 0 corresponds to DEAP trial 1 in the original dataset.

        Parameters
        ----------
        subject_id : str
            Subject identifier, e.g. ``'s01'`` or ``'S01'`` (case-insensitive).

        Returns
        -------
        dict with keys:
            subject_id : str
                Normalised subject ID (lowercase), e.g. ``'s01'``.
            eeg : ndarray, shape (40, 32, 8064)
                EEG-only data (first 32 of the 40 channels).
            ratings : ndarray, shape (40, 4)
                Self-assessment ratings: [valence, arousal, dominance, liking].
            trial_ids : list of int
                Zero-based trial indices [0, 1, …, 39].
            n_total_channels : int
                Total channel count in the original file (40).

        Raises
        ------
        FileNotFoundError
            If the expected ``.dat`` file is not found.
        ValueError
            If the file dimensions do not match the expected DEAP format.
        RuntimeError
            If the file cannot be unpickled or is otherwise unreadable.
        """
        subject_id = subject_id.lower()
        filename = f"{subject_id}.dat"
        filepath = os.path.join(self._data_dir, filename)

        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"DEAP file not found: '{filepath}'. "
                f"Subject '{subject_id}' may not be in this dataset subset."
            )

        raw = self._load_pickle(filepath)
        data_arr, labels_arr = self._extract_arrays(raw, filepath)
        self._validate_dimensions(data_arr, labels_arr, filepath)

        eeg = data_arr[:, :EXPECTED_EEG_CHANNELS, :]  # shape (40, 32, 8064)
        trial_ids = list(range(EXPECTED_TRIALS))        # [0 … 39]

        return {
            "subject_id": subject_id,
            "eeg": eeg,
            "ratings": labels_arr,
            "trial_ids": trial_ids,
            "n_total_channels": EXPECTED_TOTAL_CHANNELS,
        }

    def load_all_subjects(self) -> List[Dict[str, Any]]:
        """
        Load every discovered subject and return a list of records.

        Returns
        -------
        list of dict
            One record per subject (see :meth:`load_subject` for key schema).

        Raises
        ------
        FileNotFoundError, ValueError, RuntimeError
            Propagated from :meth:`load_subject` for the offending file.
        """
        return [self.load_subject(sid) for sid in self.discover_subjects()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pickle(filepath: str) -> Any:
        """Unpickle a DEAP .dat file.  Raises RuntimeError on failure.

        Notes
        -----
        DEAP preprocessed files were pickled with an older NumPy version that
        used legacy dtype alignment descriptors.  NumPy ≥ 2.4 emits a
        ``VisibleDeprecationWarning`` when these descriptors are encountered
        during unpickling.  This is a known compatibility issue in the DEAP
        file format itself and is suppressed here to keep output clean.
        """
        import warnings
        try:
            with open(filepath, "rb") as f:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    # VisibleDeprecationWarning was moved to numpy.exceptions in NumPy 2.0
                    try:
                        from numpy.exceptions import VisibleDeprecationWarning as _VDW
                        warnings.simplefilter("ignore", _VDW)
                    except ImportError:
                        pass  # Older NumPy: already covered by DeprecationWarning above
                    return pickle.load(f, encoding="latin1")
        except (pickle.UnpicklingError, EOFError, Exception) as exc:
            raise RuntimeError(
                f"Failed to unpickle DEAP file '{filepath}': {exc}"
            ) from exc

    @staticmethod
    def _extract_arrays(
        raw: Any, filepath: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract ``data`` and ``labels`` arrays from the unpickled object.

        DEAP preprocessed files are either a plain dict or a numpy recarray
        with keys ``'data'`` and ``'labels'``.
        """
        if isinstance(raw, dict):
            if "data" not in raw or "labels" not in raw:
                raise ValueError(
                    f"DEAP file '{filepath}' is missing expected keys "
                    f"'data' and/or 'labels'. Found keys: {list(raw.keys())}"
                )
            data_arr = np.asarray(raw["data"], dtype=np.float32)
            labels_arr = np.asarray(raw["labels"], dtype=np.float32)
        else:
            # Some DEAP distributions wrap in a recarray
            try:
                data_arr = np.asarray(raw["data"], dtype=np.float32)
                labels_arr = np.asarray(raw["labels"], dtype=np.float32)
            except (KeyError, TypeError, IndexError) as exc:
                raise ValueError(
                    f"Cannot extract 'data'/'labels' from DEAP file "
                    f"'{filepath}': {exc}"
                ) from exc
        return data_arr, labels_arr

    @staticmethod
    def _validate_dimensions(
        data_arr: np.ndarray,
        labels_arr: np.ndarray,
        filepath: str,
    ) -> None:
        """
        Assert that data and labels match the canonical DEAP dimensions.

        Expected:
            data   : (40, 40, 8064)
            labels : (40, 4)

        Raises
        ------
        ValueError
            If actual shapes do not match expectations.
        """
        expected_data_shape = (
            EXPECTED_TRIALS,
            EXPECTED_TOTAL_CHANNELS,
            EXPECTED_SAMPLES,
        )
        expected_labels_shape = (EXPECTED_TRIALS, EXPECTED_RATINGS)

        if data_arr.shape != expected_data_shape:
            raise ValueError(
                f"DEAP file '{filepath}': data shape mismatch. "
                f"Expected {expected_data_shape}, got {data_arr.shape}. "
                "Ensure this is a standard DEAP preprocessed file."
            )
        if labels_arr.shape != expected_labels_shape:
            raise ValueError(
                f"DEAP file '{filepath}': labels shape mismatch. "
                f"Expected {expected_labels_shape}, got {labels_arr.shape}."
            )
