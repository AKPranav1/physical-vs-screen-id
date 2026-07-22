"""
schemas.py

Request/response contracts for the API. Kept separate from main.py so the
pipeline modules can import these without importing FastAPI.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class BurstFrame(BaseModel):
    """One frame from the capture burst."""
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG, data-URI prefix optional")
    timestamp_ms: float = Field(..., description="Milliseconds since burst start")
    screen_flash_state: Optional[Literal["off", "white", "unknown"]] = Field(
        default="unknown",
        description="What the client's own display was showing at capture time, "
                    "used to correlate against the flash-challenge signal."
    )


class VerifyDocumentRequest(BaseModel):
    frames: List[BurstFrame]
    session_id: Optional[str] = None


class SignalScore(BaseModel):
    name: str
    score: float                    # 0..1, 1 = looks physical / genuine
    weight: float
    fired: bool                     # whether this signal had enough evidence to contribute meaningfully
    detail: Optional[str] = None


class DocTypeResult(BaseModel):
    document_type: str
    confidence: float
    source_layer: str                # "mrz" | "barcode" | "ocr_keyword" | "clip_zero_shot" | "vlm_tiebreak"
    raw_fields: Optional[dict] = None


class PadResult(BaseModel):
    verdict: Literal["physical", "screen_recapture", "manual_review"]
    fused_score: float                # 0..1
    signals: List[SignalScore]
    clip_vote_score: float
    vlm_used: bool
    vlm_reasoning: Optional[str] = None


class VerifyDocumentResponse(BaseModel):
    session_id: Optional[str]
    document: DocTypeResult
    presentation: PadResult
    stage_timings_ms: dict
