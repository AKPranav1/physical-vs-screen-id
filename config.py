"""
config.py

All tunable thresholds / weights / model names live here, ONE place.
Nothing in the pipeline modules should hardcode a magic number — if you find
yourself calibrating against real captures, this is the only file you edit.
"""

import os
import platform

# ---------------------------------------------------------------------------
# Windows: point pytesseract at the Tesseract binary explicitly, since it
# usually isn't on PATH unless you added it during install. Adjust this path
# to match wherever the UB-Mannheim installer put it on your machine.
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    _default_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default_tesseract_path):
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = _default_tesseract_path
        except ImportError:
            pass

# ---------------------------------------------------------------------------
# Device / model selection
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("ID_VERIFY_DEVICE", "cuda")  # falls back to cpu automatically if unavailable

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"   # open weights, no dataset needed

VLM_MODEL_NAME = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"             # pin a revision so behavior doesn't drift under you

# ---------------------------------------------------------------------------
# Layer 0 — deterministic parsing
# ---------------------------------------------------------------------------
MRZ_MIN_CONFIDENCE = 0.85     # passporteye's own OCR confidence before we trust the MRZ checksum result
BARCODE_REQUIRE_AAMVA_FIELDS = ["DAC", "DBB", "DCS"]  # first name / DOB / last name — sanity check the decode isn't garbage

# ---------------------------------------------------------------------------
# Layer 1 — OCR keyword classification (docTR)
# ---------------------------------------------------------------------------
# config.py - Update Layer 1

OCR_KEYWORDS = {
    "passport": [
        "passport", "passeport", "pasaporte", "reisepass", 
        "republic of india"
    ],
    "drivers_license": [
        "driver license", "driving licence", "driver's license",
        "transport department", "union of india"
    ],
    "national_id": [
        # Removed generic "identity card" from this list.
        # Now strictly limited to actual Indian Government IDs.
        "aadhaar", "unique identification authority of india", 
        "election commission of india", "elector's photo identity card", "voter",
        "income tax department", "permanent account number"
    ],
    "other_id_document": [
        # Explicitly route generic badges here
        "identity card", "student", "employee", "university", "academy", "college","P R N","blood group", "valid upto", "prn"
    ],
}
# We only need 1 strong keyword match to confidently classify the document
OCR_MIN_KEYWORD_HITS = 1 

# ---------------------------------------------------------------------------
# Layer 2 — CLIP zero-shot classification
# ---------------------------------------------------------------------------
DOC_TYPE_PROMPTS = {
    
    "passport": [
        "a photo of a passport identity document", 
        "a passport booklet photo page",
        "a digital specimen passport page",  # Add this to help CLIP identify the watermark text
        "a photo of a British Overseas Territories passport" 
    ],
    "drivers_license": [
        "a photo of a driver's license card", 
        "a driving licence plastic card",
        "a photo of an Indian driving licence"
    ],
    "national_id": [
        "a photo of a national identity card", 
        "a government issued ID card",
        "a photo of an Indian Aadhaar card",
        "a photo of an Indian Voter ID card",
        "a photo of a PAN card"
    ],
    "other_id_document": [
        "a photo of an official identification document",
        "a regular ID card or badge"
    ],
}
# Top prompt score must beat runner-up by this much to accept
CLIP_DOC_TYPE_MIN_MARGIN = 0.03 # top prompt score must beat runner-up by this much to accept, else -> escalate

# ---------------------------------------------------------------------------
# Layer 3 — VLM tie-breaker
# ---------------------------------------------------------------------------
# Only invoked when layers 0-2 disagree or all report low confidence.
VLM_MAX_NEW_TOKENS = 40

# ---------------------------------------------------------------------------
# Presentation Attack Detection (physical vs. screen/photo) — signal weights
# ---------------------------------------------------------------------------
# Weights reflect real-world reliability discussed during design: flash-challenge
# and bezel/geometry are strong and PPI-independent; moire is a bonus-only signal
# because modern high-PPI phone screens photographed by a webcam at normal
# distance frequently DON'T alias into visible moire (it gets smoothly
# downsampled instead) — see README "Why moire is weighted low" section.
PAD_SIGNAL_WEIGHTS = {
    "flash_challenge": 0.30,
    "bezel_geometry": 0.10,
    "specular_reflection": 0.15,
    "color_whitepoint": 0.12,
    "micro_parallax": 0.13,
    "moire_fft": 0.15,        # intentionally low weight, see rationale above
    "texture_lbp": 0.05,
}
assert abs(sum(PAD_SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6

PAD_CLIP_PROMPTS = {
    "physical": ["a physical ID card held in a hand", "a printed paper document held up to a camera"],
    "screen_recapture": [
        "a phone screen displaying an ID document",
        "a computer monitor showing a photo of an ID card",
    ],
}

# Decision bands on the final fused PAD score (0 = certainly screen/photo, 1 = certainly physical)
PAD_PASS_THRESHOLD = 0.68
PAD_FAIL_THRESHOLD = 0.37
# anything strictly between FAIL and PASS thresholds -> MANUAL_REVIEW, never guessed

# Weight given to the CLIP zero-shot secondary vote vs. the classical fusion score
PAD_CLIP_VOTE_WEIGHT = 0.20   # final = 0.80 * classical_fusion + 0.20 * clip_vote

# ---------------------------------------------------------------------------
# Burst capture requirements
# ---------------------------------------------------------------------------
MIN_BURST_FRAMES = 8
MAX_BURST_FRAMES = 40
FLASH_CHALLENGE_REQUIRED = False   # if True, reject bursts that don't include a flash pulse

# ---------------------------------------------------------------------------
# Logging / calibration
# ---------------------------------------------------------------------------
CALIBRATION_LOG_MODE = os.environ.get("CALIBRATION_LOG_MODE", "csv")   # "stdout" | "csv" | "both"
CALIBRATION_LOG_PATH = os.path.join(os.path.dirname(__file__), "calibration", "signal_log.csv")
