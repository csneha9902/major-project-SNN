"""
DEAP dataset configuration.

Reads the DEAP_DATA_DIR environment variable that points to the local directory
containing preprocessed DEAP .dat files (s01.dat … s32.dat subset).

The dataset must be supplied by the researcher. It is never downloaded automatically
and is never committed to the repository.

Environment variables
---------------------
DEAP_DATA_DIR : str
    Absolute path to the directory containing DEAP .dat files.
    Example: /Users/sneha/Datasets/DEAP

Label ordering in DEAP ratings array (columns 0–3)
----------------------------------------------------
  0 : valence    (continuous 1–9)
  1 : arousal    (continuous 1–9)  ← research target for Task 1
  2 : dominance  (continuous 1–9)
  3 : liking     (continuous 1–9)

Arousal binary label convention
---------------------------------
  LOW_AROUSAL  : arousal rating ≤ AROUSAL_THRESHOLD
  HIGH_AROUSAL : arousal rating >  AROUSAL_THRESHOLD
"""

import os

# ---------------------------------------------------------------------------
# Label class names — do NOT rename to Stress / Focused / Neutral
# ---------------------------------------------------------------------------
LOW_AROUSAL: str = "LOW_AROUSAL"
HIGH_AROUSAL: str = "HIGH_AROUSAL"

# ---------------------------------------------------------------------------
# Default arousal binarisation threshold (configurable at runtime)
# ---------------------------------------------------------------------------
DEFAULT_AROUSAL_THRESHOLD: float = 5.0

# ---------------------------------------------------------------------------
# DEAP file format constants (preprocessed Python pickle format)
# ---------------------------------------------------------------------------
EXPECTED_TRIALS: int = 40
EXPECTED_TOTAL_CHANNELS: int = 40   # 32 EEG + 8 peripheral
EXPECTED_EEG_CHANNELS: int = 32
EXPECTED_SAMPLES: int = 8064        # 63 s × 128 Hz
EXPECTED_RATINGS: int = 4           # valence, arousal, dominance, liking

# Column indices in the ratings array
RATING_VALENCE_IDX: int = 0
RATING_AROUSAL_IDX: int = 1
RATING_DOMINANCE_IDX: int = 2
RATING_LIKING_IDX: int = 3

# Subject file naming pattern
SUBJECT_FILE_PATTERN: str = "s{:02d}.dat"   # e.g. s01.dat, s22.dat


def get_deap_data_dir() -> str:
    """
    Return the validated path to the DEAP dataset directory.

    Reads the ``DEAP_DATA_DIR`` environment variable.  Raises
    ``EnvironmentError`` if the variable is not set and ``FileNotFoundError``
    if the path does not exist on disk.

    Returns
    -------
    str
        Absolute path to the DEAP data directory.

    Raises
    ------
    EnvironmentError
        If ``DEAP_DATA_DIR`` environment variable is not set.
    FileNotFoundError
        If the directory named by ``DEAP_DATA_DIR`` does not exist.
    """
    data_dir = os.environ.get("DEAP_DATA_DIR", "").strip()
    if not data_dir:
        raise EnvironmentError(
            "DEAP_DATA_DIR environment variable is not set. "
            "Set it to the absolute path of the directory containing "
            "DEAP .dat files, e.g.:\n"
            "  export DEAP_DATA_DIR=/path/to/DEAP"
        )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DEAP_DATA_DIR='{data_dir}' does not exist or is not a directory. "
            "Supply the DEAP dataset locally — it is not downloaded automatically."
        )
    return data_dir
