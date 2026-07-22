"""
calibration/review_calibration.py

Run real captures through the API first (both genuine physical-document
captures and deliberate screen-recapture attempts), which populates
calibration/signal_log.csv via utils/logging_calibration.py.

Then run this script to see per-signal separation between the two groups,
which tells you whether a signal is actually pulling its weight or should
have its weight/threshold adjusted in config.py.

Usage:
    python calibration/review_calibration.py --known-physical-sessions s1,s2,s3 \\
                                              --known-screen-sessions s4,s5,s6
"""

import argparse
import csv
import os
import sys

SIGNAL_COLUMNS = [
    "flash_challenge", "bezel_geometry", "specular_reflection",
    "color_whitepoint", "micro_parallax", "moire_fft", "texture_lbp",
    "clip_vote_score", "fused_score",
]

LOG_PATH = os.path.join(os.path.dirname(__file__), "signal_log.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-physical-sessions", default="", help="comma-separated session_ids known to be genuine physical docs")
    parser.add_argument("--known-screen-sessions", default="", help="comma-separated session_ids known to be screen recaptures")
    args = parser.parse_args()

    physical_ids = set(args.known_physical_sessions.split(",")) if args.known_physical_sessions else set()
    screen_ids = set(args.known_screen_sessions.split(",")) if args.known_screen_sessions else set()

    if not os.path.exists(LOG_PATH):
        print(f"No calibration log found at {LOG_PATH} yet — run some captures through the API first.")
        sys.exit(1)

    physical_rows, screen_rows = [], []
    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["session_id"] in physical_ids:
                physical_rows.append(row)
            elif row["session_id"] in screen_ids:
                screen_rows.append(row)

    if not physical_rows or not screen_rows:
        print("Need at least one labeled session in each group to compute separation.")
        sys.exit(1)

    print(f"{'signal':<22} {'physical mean':>14} {'screen mean':>14} {'separation':>12}")
    print("-" * 66)
    for col in SIGNAL_COLUMNS:
        try:
            p_vals = [float(r[col]) for r in physical_rows if r.get(col)]
            s_vals = [float(r[col]) for r in screen_rows if r.get(col)]
        except (ValueError, KeyError):
            continue
        if not p_vals or not s_vals:
            continue
        p_mean = sum(p_vals) / len(p_vals)
        s_mean = sum(s_vals) / len(s_vals)
        print(f"{col:<22} {p_mean:>14.3f} {s_mean:>14.3f} {abs(p_mean - s_mean):>12.3f}")

    print("\nSignals with small separation are candidates to re-weight (lower) in config.PAD_SIGNAL_WEIGHTS.")
    print("Signals with large, consistent separation are candidates to weight higher.")


if __name__ == "__main__":
    main()
