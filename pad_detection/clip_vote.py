"""
pad_detection/clip_vote.py

Secondary, fast vote alongside the classical-CV fusion. Reuses the SAME
CLIP embedding computed for document-type classification on the same
representative frame (see main.py) -- no extra forward pass.
"""

import torch
from config import PAD_CLIP_PROMPTS
from utils.clip_engine import score_against_prompt_groups


def score_clip_vote(cached_embedding: torch.Tensor) -> dict:
    result = score_against_prompt_groups(cached_embedding, PAD_CLIP_PROMPTS)
    confidence = result["confidence"]

    physical_conf = confidence.get("physical", 0.5)
    return {
        "score": float(physical_conf),   # already 0..1, 1 = physical
        "detail": f"CLIP zero-shot vote: physical={physical_conf:.3f}, "
                  f"screen={confidence.get('screen_recapture', 0.0):.3f}",
    }
