"""
doc_classification/vlm_tiebreak.py

Layer 3 — last resort. Only invoked when Layers 0-2 disagree or all report
low confidence, which should be a small minority of requests. The model is
lazy-loaded on first use so it costs you nothing (no VRAM, no load time) on
the common fast path.

Also doubles as the PAD tie-breaker (pad_detection/vlm_pad_tiebreak.py reuses
the same loaded model) so you only ever pay the VLM's load cost once.
"""

from typing import Optional
import threading
import numpy as np
import cv2
from PIL import Image

from config import VLM_MODEL_NAME, VLM_REVISION, VLM_MAX_NEW_TOKENS, DEVICE

_model = None
_tokenizer = None
_lock = threading.Lock()


def _get_model():
    global _model, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32

                model = AutoModelForCausalLM.from_pretrained(
                    VLM_MODEL_NAME, revision=VLM_REVISION,
                    trust_remote_code=True, torch_dtype=dtype,
                ).to(device).eval()
                tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL_NAME, revision=VLM_REVISION)

                _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def ask_document_type(image: np.ndarray) -> Optional[dict]:
    """Free-text tie-break for document type. Parsed leniently against known labels."""
    model, tokenizer = _get_model()
    pil_img = _to_pil(image)

    prompt = (
        "What type of identity document is shown in this image? "
        "Answer with one of: passport, driver's license, national ID card, "
        "or describe briefly if it is something else."
    )
    enc_image = model.encode_image(pil_img)
    answer = model.answer_question(enc_image, prompt, tokenizer, max_new_tokens=VLM_MAX_NEW_TOKENS)
    answer_lower = answer.lower()

    if "passport" in answer_lower:
        doc_type = "passport"
    elif "driver" in answer_lower or "driving" in answer_lower:
        doc_type = "drivers_license"
    elif "national" in answer_lower or "identity card" in answer_lower or "id card" in answer_lower:
        doc_type = "national_id"
    else:
        doc_type = "other_id_document"

    return {
        "document_type": doc_type,
        "confidence": 0.75,   # VLM free-text answers get a moderate, not high, confidence by design
        "source_layer": "vlm_tiebreak",
        "raw_fields": {"raw_answer": answer},
    }


def ask_physical_or_screen(image: np.ndarray) -> dict:
    """Free-text tie-break for the PAD decision. Returns a score in [0,1], 1=physical."""
    model, tokenizer = _get_model()
    pil_img = _to_pil(image)

    prompt = (
        "Look closely at this image. Is it a photo of a physical ID document held in "
        "someone's hand, or is it a photo of a phone or computer screen that is displaying "
        "an image of an ID document? Answer with 'physical' or 'screen'."
    )
    enc_image = model.encode_image(pil_img)
    answer = model.answer_question(enc_image, prompt, tokenizer, max_new_tokens=VLM_MAX_NEW_TOKENS)
    answer_lower = answer.lower()

    if "screen" in answer_lower or "phone" in answer_lower or "monitor" in answer_lower:
        score = 0.15
    elif "physical" in answer_lower or "hand" in answer_lower or "paper" in answer_lower or "card" in answer_lower:
        score = 0.85
    else:
        score = 0.5   # genuinely couldn't parse an answer -> neutral, let other signals decide

    return {"score": score, "raw_answer": answer}
