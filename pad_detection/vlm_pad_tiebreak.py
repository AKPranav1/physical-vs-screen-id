"""
pad_detection/vlm_pad_tiebreak.py

Thin wrapper so pad_detection/pipeline.py doesn't need to import from
doc_classification directly -- both point at the same lazy-loaded Moondream2
instance under the hood, so the model is only ever loaded once regardless
of which pipeline triggers it first.
"""

import numpy as np
from doc_classification.vlm_tiebreak import ask_physical_or_screen


def run_vlm_pad_tiebreak(frame: np.ndarray) -> dict:
    return ask_physical_or_screen(frame)
