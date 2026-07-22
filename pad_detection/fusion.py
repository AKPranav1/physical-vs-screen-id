"""
pad_detection/fusion.py

Combines all classical signals + the CLIP secondary vote into one fused
score, with an explicit MANUAL_REVIEW band for anything ambiguous. This is
a deliberately simple, interpretable weighted-sum fusion (not a trained
classifier) -- see README "Why interpretable fusion over a trained model"
for the reasoning.

Only escalates to the (slow) VLM tie-break when the fused score AND the
CLIP vote disagree with each other, or both land inside the ambiguous band.
"""

from typing import List
from config import (
    PAD_SIGNAL_WEIGHTS, PAD_PASS_THRESHOLD, PAD_FAIL_THRESHOLD, PAD_CLIP_VOTE_WEIGHT
)


def fuse_signals(signal_scores: dict) -> dict:
    """
    signal_scores: {signal_name: {"score": float, "fired": bool, "detail": str}}
    Returns {"classical_fusion_score": float, "signals": [SignalScore-shaped dicts]}
    """
    weighted_sum = 0.0
    total_weight_used = 0.0
    signal_list = []

    for name, weight in PAD_SIGNAL_WEIGHTS.items():
        sig = signal_scores.get(name)
        if sig is None:
            continue

        # A signal that didn't fire (not enough evidence) contributes at
        # neutral (0.5) rather than dragging the fusion score toward either
        # side -- it should not be treated the same as a confident 0 or 1.
        effective_score = sig["score"] if sig["fired"] else 0.5
        weighted_sum += effective_score * weight
        total_weight_used += weight

        signal_list.append({
            "name": name,
            "score": round(sig["score"], 4),
            "weight": weight,
            "fired": sig["fired"],
            "detail": sig.get("detail"),
        })

    classical_score = weighted_sum / total_weight_used if total_weight_used > 0 else 0.5

    return {"classical_fusion_score": classical_score, "signals": signal_list}


def combine_with_clip_vote(classical_score: float, clip_vote_score: float) -> float:
    return (1 - PAD_CLIP_VOTE_WEIGHT) * classical_score + PAD_CLIP_VOTE_WEIGHT * clip_vote_score


def needs_vlm_tiebreak(fused_score: float, classical_score: float, clip_vote_score: float) -> bool:
    """
    Escalate to the VLM only when genuinely ambiguous:
      - the fused score itself lands in the manual-review band, OR
      - classical fusion and the CLIP vote meaningfully disagree with each other
    """
    in_review_band = PAD_FAIL_THRESHOLD < fused_score < PAD_PASS_THRESHOLD
    disagreement = abs(classical_score - clip_vote_score) > 0.35
    return in_review_band or disagreement


def final_verdict(fused_score: float) -> str:
    if fused_score >= PAD_PASS_THRESHOLD:
        return "physical"
    if fused_score <= PAD_FAIL_THRESHOLD:
        return "screen_recapture"
    return "manual_review"
