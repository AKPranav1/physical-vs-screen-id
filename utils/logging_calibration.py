"""
utils/logging_calibration.py

Logs every raw signal score per request so PAD_SIGNAL_WEIGHTS and the
PASS/FAIL thresholds in config.py can be tuned against real captures instead
of guessed. This is what lets you calibrate WITHOUT building a training
dataset — just run real captures through the system, look at this CSV, and
adjust config.py by hand.
"""

import csv
import os
import time
from config import CALIBRATION_LOG_MODE, CALIBRATION_LOG_PATH

_CSV_HEADER = [
    "timestamp", "session_id",
    "flash_challenge", "bezel_geometry", "specular_reflection",
    "color_whitepoint", "micro_parallax", "moire_fft", "texture_lbp",
    "clip_vote_score", "fused_score", "verdict",
    "doc_type", "doc_type_source", "doc_type_confidence",
]


def _ensure_csv_exists():
    os.makedirs(os.path.dirname(CALIBRATION_LOG_PATH), exist_ok=True)
    if not os.path.exists(CALIBRATION_LOG_PATH):
        with open(CALIBRATION_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def log_calibration_row(session_id: str, row: dict) -> None:
    full_row = {"timestamp": time.time(), "session_id": session_id, **row}

    if CALIBRATION_LOG_MODE in ("stdout", "both"):
        print(f"[calibration] {full_row}")

    if CALIBRATION_LOG_MODE in ("csv", "both"):
        _ensure_csv_exists()
        with open(CALIBRATION_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER, extrasaction="ignore")
            writer.writerow(full_row)
