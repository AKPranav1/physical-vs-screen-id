"""
doc_classification/pipeline.py

Orchestrates the 4-layer document-type cascade:
  0. MRZ / barcode (deterministic)
  1. docTR OCR keywords
  2. CLIP zero-shot
  3. VLM tie-break (rare)

Each layer only runs if the previous ones didn't produce a confident answer.
This keeps the common-case latency low (most documents resolve at layer 0/1/2)
while still handling arbitrary/unseen document types via layer 2's open
vocabulary and layer 3's free-text fallback.
"""

from typing import List, Optional
import numpy as np
import torch

from doc_classification.mrz_barcode import try_deterministic_layers
from doc_classification.ocr_doctr import classify_by_ocr_keywords
from doc_classification.clip_zero_shot import classify_by_clip
from doc_classification.vlm_tiebreak import ask_document_type


def sharpest_frame(frames: List[np.ndarray]) -> np.ndarray:
    """Pick the least-blurry frame (highest Laplacian variance) for OCR/MRZ attempts."""
    import cv2
    best_idx, best_score = 0, -1.0
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_idx, best_score = i, score
    return frames[best_idx]


def classify_document(frames: List[np.ndarray], cached_clip_embedding: Optional[torch.Tensor] = None) -> dict:
    """
    frames: list of decoded BGR frames from the burst.
    cached_clip_embedding: optional, reused from the PAD pipeline's own CLIP
    call on the same representative frame, to avoid a duplicate forward pass.
    """
    best_frame = sharpest_frame(frames)

    # Layer 0
    result = try_deterministic_layers(frames)
    if result:
        return result

    # Layer 1
    result = classify_by_ocr_keywords(best_frame)
    if result:
        return result

    # Layer 2
    result = classify_by_clip(best_frame, cached_embedding=cached_clip_embedding)
    if result:
        return result

    # Layer 3 — rare tie-break
    result = ask_document_type(best_frame)
    return result
