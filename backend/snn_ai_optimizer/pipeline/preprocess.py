# backend/snn_ai_optimizer/pipeline/preprocess.py

from pathlib import Path
import json

from snn_ai_optimizer.utils.logger import (
    create_run_folder,
    save_metrics,
    save_text_log,
)

# EEG loader
from snn_ai_optimizer.datasets.eeg_loader import (
    load_sample_eeg,
    preprocess_eeg,
)

import os

ENABLE_MRI = os.environ.get("ENABLE_MRI", "false").lower() in ("true", "1")


def main():
    """Simple entry: run full preprocess and mark status."""
    print("[Preprocess] Starting EEG preprocessing...")
    run_path = create_run_folder("preprocess", overwrite=True)

    steps = [
        "Loading EEG",
        "Filtering/Feature extraction (EEG)",
        "Write outputs",
    ]
    for s in steps:
        print(f"[Preprocess] {s}")
        save_text_log(run_path, f"[Preprocess] {s}")

    # Run the real pipeline
    preprocess_run(run_path=run_path)

    # Overall status metric
    save_metrics(run_path, "preprocess", {"status": "completed", "steps": steps})
    print("[Preprocess] Finished EEG preprocessing.")


def preprocess_run(run_path=None):
    """
    Create EEG band-power features and save them under results/preprocess/.
    MRI processing is decoupled per AGENTS.md Rule 13 unless ENABLE_MRI=1 is explicitly set.
    """
    if run_path is None:
        run_path = create_run_folder("preprocess")
    out_dir = Path("results/preprocess")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- EEG ----------------
    eeg_ok = False
    try:
        raw = load_sample_eeg()
        eeg_feats = preprocess_eeg(raw)
        (out_dir / "eeg_features.json").write_text(
            json.dumps({"features": eeg_feats}, indent=2),
            encoding="utf-8",
        )
        save_text_log(run_path, f"[Preprocess] EEG features saved ({len(eeg_feats)} dims)")
        save_metrics(run_path, "preprocess_eeg", {"dims": len(eeg_feats), "ok": True})
        eeg_ok = True
    except Exception as e:
        save_text_log(run_path, f"[Preprocess] EEG preprocessing info: {e}")
        save_metrics(run_path, "preprocess_eeg", {"ok": False, "error": str(e)})

    # ---------------- MRI (Decoupled per AGENTS.md Rule 13) ----------------
    if ENABLE_MRI:
        try:
            from snn_ai_optimizer.datasets.mri_loader import load_sample_mri, preprocess_mri
            img = load_sample_mri()
            mri_feats = preprocess_mri(img)
            (out_dir / "mri_features.json").write_text(
                json.dumps({"features": mri_feats}, indent=2),
                encoding="utf-8",
            )
            save_text_log(run_path, f"[Preprocess] MRI features saved ({len(mri_feats)} dims)")
            save_metrics(run_path, "preprocess_mri", {"dims": len(mri_feats), "ok": True})
        except Exception as e:
            save_text_log(run_path, f"[Preprocess] MRI failed: {e}")
            save_metrics(run_path, "preprocess_mri", {"ok": False, "error": str(e)})

    # Final status log
    save_text_log(
        run_path,
        f"[Preprocess] Done. EEG ok={eeg_ok}. Artifacts at {out_dir}",
    )


if __name__ == "__main__":
    main()
