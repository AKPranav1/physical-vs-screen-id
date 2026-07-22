"""
doc_classification/mrz_barcode.py

Layer 0 of the document-type cascade: deterministic parsing.

This layer has NO ML uncertainty at all. If an MRZ checksum validates, or a
PDF417 barcode decodes into well-formed AAMVA fields, we know the document
type with near-100% certainty and every downstream layer is skipped.

Requires system packages: tesseract-ocr (for passporteye), libzbar0 (for pyzbar)
"""

from typing import Optional
import numpy as np
import tempfile
import os
import cv2

from config import MRZ_MIN_CONFIDENCE, BARCODE_REQUIRE_AAMVA_FIELDS


def try_parse_mrz(image: np.ndarray) -> Optional[dict]:
    """
    Attempts MRZ extraction via PassportEye. Returns a result dict only if
    the MRZ was found AND its internal checksum validation passed AND OCR
    confidence clears MRZ_MIN_CONFIDENCE. Otherwise returns None (do not guess).
    """
    try:
        from passporteye import read_mrz
    except (ImportError, OSError):
        return None

    # passporteye needs a file path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name

    try:
        mrz = read_mrz(tmp_path)
    except Exception:
        return None
    finally:
        os.unlink(tmp_path)

    if mrz is None:
        return None

    fields = mrz.to_dict()
    valid_score = fields.get("valid_score", 0) / 100.0   # passporteye reports 0-100
    checksum_ok = bool(fields.get("valid_composite", False)) or bool(fields.get("valid_number", False))

    if valid_score < MRZ_MIN_CONFIDENCE or not checksum_ok:
        return None

    mrz_type = fields.get("mrz_type", "")
    # TD3 = passport format, TD1/TD2 = national ID / residence permit formats
    doc_type = "passport" if mrz_type == "TD3" else "national_id"

    return {
        "document_type": doc_type,
        "confidence": min(0.99, valid_score),
        "source_layer": "mrz",
        "raw_fields": {
            "mrz_type": mrz_type,
            "country": fields.get("country"),
            "nationality": fields.get("nationality"),
        },
    }


def try_parse_barcode(image: np.ndarray) -> Optional[dict]:
    """
    Attempts PDF417 barcode decode via zxing-cpp (self-contained wheel, no
    external native DLL dependencies -- more reliable across platforms than
    pyzbar/libzbar). If it decodes and contains a plausible subset of AAMVA
    field codes, this is a US/CA driver's license with near-certainty.
    """
    try:
        import zxingcpp
    except (ImportError, OSError):
        return None

    try:
        results = zxingcpp.read_barcodes(image)
    except Exception:
        return None

    for r in results:
        if "PDF417" not in str(r.format):
            continue
        try:
            text = r.text
        except Exception:
            continue
        if not text:
            continue

        hits = sum(1 for field in BARCODE_REQUIRE_AAMVA_FIELDS if field in text)
        if hits == 0:
            continue   # decoded *something* but it doesn't look like AAMVA — don't claim it

        return {
            "document_type": "drivers_license",
            "confidence": 0.97,
            "source_layer": "barcode",
            "raw_fields": {"aamva_field_hits": hits},
        }

    return None


def try_deterministic_layers(frames: list) -> Optional[dict]:
    """
    Tries MRZ then barcode across the burst (uses the sharpest frame first —
    caller should pass frames pre-sorted by sharpness for best odds).
    Returns the first confident hit, or None.
    """
    for img in frames:
        result = try_parse_mrz(img)
        if result:
            return result

    for img in frames:
        result = try_parse_barcode(img)
        if result:
            return result

    return None
