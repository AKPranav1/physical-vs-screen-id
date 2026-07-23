"""
doc_classification/ocr_doctr.py

Layer 1 of the document-type cascade: OCR-based keyword classification via
docTR (Mindee, Apache-2.0, pretrained — no training data needed).

Fires when there is no MRZ/barcode but the document has readable printed
headers ("DRIVER LICENSE", "PASSPORT", "IDENTITY CARD", ...). This covers
the large middle ground of documents that are real and legible but don't
carry a machine-readable zone.
"""

from typing import Optional
import numpy as np
import threading

from config import OCR_KEYWORDS, OCR_MIN_KEYWORD_HITS

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from doctr.models import ocr_predictor
                _model = ocr_predictor(pretrained=True)
    return _model


def _extract_text(image: np.ndarray) -> str:
    from doctr.io import DocumentFile
    import cv2
    import tempfile
    import os

    # 1. Open the temp file, get the name, and immediately close the OS handle
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.close()

    try:
        # 2. Now that the handle is free, write the image and let docTR read it
        cv2.imwrite(tmp_path, image)
        doc = DocumentFile.from_images(tmp_path)
        result = _get_model()(doc)
        text = result.render()
    finally:
        # 3. Clean up safely
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return text.lower()


def classify_by_ocr_keywords(image: np.ndarray) -> Optional[dict]:
    """
    Returns a document-type guess based on keyword hits in OCR'd text, or
    None if no keyword set clears OCR_MIN_KEYWORD_HITS (do not guess).
    """
    try:
        text = _extract_text(image)
    except Exception:
        return None

    best_type = None
    best_hits = 0
    for doc_type, keywords in OCR_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_hits:
            best_hits = hits
            best_type = doc_type

    if best_type is None or best_hits < OCR_MIN_KEYWORD_HITS:
        return None

    # Confidence scales gently with number of distinct keyword hits, capped —
    # this is a heuristic signal, not a calibrated probability, so keep it modest.
    confidence = min(0.90, 0.6 + 0.1 * best_hits)

    return {
        "document_type": best_type,
        "confidence": confidence,
        "source_layer": "ocr_keyword",
        "raw_fields": {"keyword_hits": best_hits},
    }