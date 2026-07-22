"""
pad_detection/pipeline.py

Orchestrates presentation-attack detection (physical vs. screen/photo):

  1. Run all classical signals (parallelizable, all CPU except none here)
  2. Fuse them into classical_fusion_score
  3. Add the CLIP secondary vote (reusing the cached embedding from doc-type)
  4. If genuinely ambiguous -> escalate to VLM tie-break (rare path)
  5. Apply PASS/FAIL/MANUAL_REVIEW thresholds -- never force a guess
"""

from typing import List, Tuple, Optional
import numpy as np
import torch

from pad_detection.flash_challenge import score_flash_challenge
from pad_detection.bezel_geometry import score_bezel_geometry
from pad_detection.specular_reflection import score_specular_reflection
from pad_detection.color_whitepoint import score_color_whitepoint
from pad_detection.micro_parallax import score_micro_parallax
from pad_detection.moire_fft import score_moire_fft
from pad_detection.texture_lbp import score_texture_lbp
from pad_detection.clip_vote import score_clip_vote
from pad_detection.vlm_pad_tiebreak import run_vlm_pad_tiebreak
from pad_detection.fusion import (
    fuse_signals, combine_with_clip_vote, needs_vlm_tiebreak, final_verdict
)


def run_pad_pipeline(
    frames_with_state: List[Tuple[np.ndarray, float, str]],
    representative_frame: np.ndarray,
    cached_clip_embedding: Optional[torch.Tensor] = None,
) -> dict:
    plain_frames = [f for f, _, _ in frames_with_state]

    signal_scores = {
        "flash_challenge": score_flash_challenge(frames_with_state),
        "bezel_geometry": score_bezel_geometry(representative_frame),
        "specular_reflection": score_specular_reflection(representative_frame),
        "color_whitepoint": score_color_whitepoint(representative_frame),
        "micro_parallax": score_micro_parallax(plain_frames),
        "moire_fft": score_moire_fft(plain_frames),
        "texture_lbp": score_texture_lbp(representative_frame),
    }

    fusion_result = fuse_signals(signal_scores)
    classical_score = fusion_result["classical_fusion_score"]

    from utils.clip_engine import embed_image
    clip_embedding = cached_clip_embedding if cached_clip_embedding is not None else embed_image(representative_frame)
    clip_vote = score_clip_vote(clip_embedding)
    clip_vote_score = clip_vote["score"]

    fused_score = combine_with_clip_vote(classical_score, clip_vote_score)

    vlm_used = False
    vlm_reasoning = None

    if needs_vlm_tiebreak(fused_score, classical_score, clip_vote_score):
        vlm_result = run_vlm_pad_tiebreak(representative_frame)
        vlm_used = True
        vlm_reasoning = vlm_result["raw_answer"]
        # Re-fuse giving the VLM a decisive-but-not-absolute weight, since it's
        # only invoked precisely because the cheaper signals couldn't agree.
        fused_score = 0.5 * fused_score + 0.5 * vlm_result["score"]

    verdict = final_verdict(fused_score)

    return {
        "verdict": verdict,
        "fused_score": round(fused_score, 4),
        "signals": fusion_result["signals"],
        "clip_vote_score": round(clip_vote_score, 4),
        "vlm_used": vlm_used,
        "vlm_reasoning": vlm_reasoning,
    }
