"""
main.py

FastAPI entrypoint. One endpoint: POST /verify-document

Request:  a burst of frames (see schemas.VerifyDocumentRequest), some of
          which are labeled with the client's own screen_flash_state so the
          flash-challenge signal can correlate against it.

Response: document type (with the layer that resolved it) + presentation
          verdict (physical / screen_recapture / manual_review), full signal
          breakdown for transparency, and stage timings.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager # Added for lifespan
import uuid

from config import MIN_BURST_FRAMES, MAX_BURST_FRAMES, FLASH_CHALLENGE_REQUIRED
from schemas import VerifyDocumentRequest, VerifyDocumentResponse, DocTypeResult, PadResult
from utils.decoding import decode_burst, assert_consistent_resolution
from utils.timing import StageTimer
from utils.logging_calibration import log_calibration_row
from utils.clip_engine import embed_image, _get_model as get_clip_model # Added CLIP model loader
from doc_classification.pipeline import classify_document, sharpest_frame
from pad_detection.pipeline import run_pad_pipeline
from doc_classification.vlm_tiebreak import _get_model as get_vlm_model # Added VLM model loader
from doc_classification.ocr_doctr import _get_model as get_doctr_model

# --- Added Lifespan function for Eager Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-loading models into memory...")
    get_clip_model() # Loads OpenCLIP
    get_doctr_model()  # Loads docTR OCR
    get_vlm_model()  # Loads Moondream2 VLM
    print("All models pre-loaded and ready!")
    yield
# -------------------------------------------------

# Attach the lifespan to the FastAPI app initialization
app = FastAPI(title="ID Document Verification API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify-document", response_model=VerifyDocumentResponse)
def verify_document(request: VerifyDocumentRequest):
    print(f"\n====== API GATEWAY CROSSED: {len(request.frames)} frames received ======\n")
    try:
        timer = StageTimer()
        session_id = request.session_id or str(uuid.uuid4())

        with timer.stage("decode_burst"):
            decoded = decode_burst(request.frames, MIN_BURST_FRAMES, MAX_BURST_FRAMES)
            plain_frames = [img for img, _, _ in decoded]
            assert_consistent_resolution(plain_frames)

        if FLASH_CHALLENGE_REQUIRED:
            states = {state for _, _, state in decoded}
            if "white" not in states or "off" not in states:
                raise HTTPException(
                    status_code=400,
                    detail="Flash-challenge is required but burst does not contain both "
                           "'white' and 'off' labeled frames"
                )

        with timer.stage("pick_representative_frame"):
            representative_frame = sharpest_frame(plain_frames)

        # Compute the CLIP embedding on the representative frame ONCE and reuse it
        # for both document-type classification (layer 2) and the PAD secondary
        # vote -- this is the single most impactful latency optimization here.
        with timer.stage("clip_embed_shared"):
            cached_embedding = embed_image(representative_frame)

        with timer.stage("classify_document"):
            doc_result = classify_document(plain_frames, cached_clip_embedding=cached_embedding)

        with timer.stage("presentation_attack_detection"):
            pad_result = run_pad_pipeline(
                decoded, representative_frame, cached_clip_embedding=cached_embedding
            )

        stage_timings = timer.as_dict()

        log_calibration_row(session_id, {
            **{s["name"]: s["score"] for s in pad_result["signals"]},
            "clip_vote_score": pad_result["clip_vote_score"],
            "fused_score": pad_result["fused_score"],
            "verdict": pad_result["verdict"],
            "doc_type": doc_result["document_type"],
            "doc_type_source": doc_result["source_layer"],
            "doc_type_confidence": doc_result["confidence"],
        })

        return VerifyDocumentResponse(
            session_id=session_id,
            document=DocTypeResult(**doc_result),
            presentation=PadResult(**pad_result),
            stage_timings_ms=stage_timings,
        )

    except Exception as e:
        import traceback
        print("CRITICAL ERROR IN ENDPOINT:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))