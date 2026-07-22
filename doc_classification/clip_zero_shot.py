"""
doc_classification/clip_zero_shot.py

Layer 2 of the document-type cascade: open-vocabulary zero-shot classification
via OpenCLIP. Fires when neither the MRZ/barcode layer nor OCR keywords gave
a confident answer (glare, unfamiliar layout, damaged text, or genuinely a
document type we haven't seen before).

This is the layer that makes "any document, no restriction" actually true —
adding a new document type means adding a text prompt to config.DOC_TYPE_PROMPTS,
never retraining anything.
"""

from typing import Optional
import numpy as np
import torch

from config import DOC_TYPE_PROMPTS, CLIP_DOC_TYPE_MIN_MARGIN
from utils.clip_engine import embed_image, score_against_prompt_groups


def classify_by_clip(image: np.ndarray, cached_embedding: Optional[torch.Tensor] = None) -> Optional[dict]:
    """
    Returns the best-matching document type by CLIP zero-shot similarity, or
    None if the top match doesn't beat the runner-up by CLIP_DOC_TYPE_MIN_MARGIN
    (ambiguous -> let the VLM tie-breaker decide instead of guessing).

    cached_embedding: pass in the already-computed image embedding (see main.py)
    so we don't run CLIP twice per frame across the doc-type and PAD pipelines.
    """
    image_emb = cached_embedding if cached_embedding is not None else embed_image(image)

    result = score_against_prompt_groups(image_emb, DOC_TYPE_PROMPTS)
    confidence = result["confidence"]

    ranked = sorted(confidence.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if (top_score - runner_up_score) < CLIP_DOC_TYPE_MIN_MARGIN:
        return None   # too ambiguous, don't force it

    return {
        "document_type": top_label,
        "confidence": round(top_score, 4),
        "source_layer": "clip_zero_shot",
        "raw_fields": {"all_scores": confidence},
    }
