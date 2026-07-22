"""
utils/clip_engine.py

Loads OpenCLIP exactly once and exposes a simple embed + zero-shot-compare
API. Both doc_classification/clip_zero_shot.py and pad_detection/clip_vote.py
reuse the SAME loaded model and the SAME per-request image embedding —
we only ever run one CLIP forward pass per frame, no matter how many
zero-shot comparisons are made against it.
"""

import threading
import numpy as np
import torch
from PIL import Image
import cv2

from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, DEVICE

_model = None
_preprocess = None
_tokenizer = None
_lock = threading.Lock()
_device = None


def _resolve_device() -> str:
    global _device
    if _device is not None:
        return _device
    if DEVICE == "cuda" and torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"
    return _device


def _get_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
                )
                model.eval().to(_resolve_device())
                tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
                _model, _preprocess, _tokenizer = model, preprocess, tokenizer
    return _model, _preprocess, _tokenizer


def embed_image(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR np.ndarray -> normalized CLIP image embedding (1, D) on the model's device."""
    model, preprocess, _ = _get_model()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(_resolve_device())

    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def embed_texts(prompts: list) -> torch.Tensor:
    """List[str] -> normalized CLIP text embeddings (N, D)."""
    model, _, tokenizer = _get_model()
    tokens = tokenizer(prompts).to(_resolve_device())

    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def score_against_prompt_groups(image_emb: torch.Tensor, prompt_groups: dict) -> dict:
    """
    prompt_groups: {label: [prompt1, prompt2, ...]}
    Returns {label: max_similarity_score} using the best-matching prompt per group,
    plus a softmax-normalized confidence distribution across groups.
    """
    all_prompts = []
    group_ranges = {}
    idx = 0
    for label, prompts in prompt_groups.items():
        group_ranges[label] = (idx, idx + len(prompts))
        all_prompts.extend(prompts)
        idx += len(prompts)

    text_emb = embed_texts(all_prompts)
    sims = (image_emb @ text_emb.T).squeeze(0)   # (num_prompts,)

    group_scores = {}
    for label, (start, end) in group_ranges.items():
        group_scores[label] = float(sims[start:end].max().item())

    # softmax over group max-scores for a normalized confidence read
    labels = list(group_scores.keys())
    logits = torch.tensor([group_scores[l] for l in labels]) * 100.0  # CLIP logit scale convention
    probs = torch.softmax(logits, dim=0)
    confidence = {labels[i]: float(probs[i].item()) for i in range(len(labels))}

    return {"raw_similarity": group_scores, "confidence": confidence}
