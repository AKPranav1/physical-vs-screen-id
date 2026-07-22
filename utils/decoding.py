"""
utils/decoding.py

Every malformed-input path here must raise HTTPException(400, ...) — never let
a bad base64 string or corrupt image bubble up as an unhandled 500.
"""

import base64
import binascii
import numpy as np
import cv2
from fastapi import HTTPException


def decode_base64_image(b64_string: str, field_name: str = "image") -> np.ndarray:
    """
    Decode a base64 string into a BGR cv2 image (np.ndarray, HxWx3, uint8).
    Raises HTTPException(400) on any failure.
    """
    if not b64_string or not isinstance(b64_string, str):
        raise HTTPException(status_code=400, detail=f"{field_name}: empty or invalid payload")

    if "," in b64_string and b64_string.strip().startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(b64_string, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"{field_name}: not valid base64 ({e})")

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail=f"{field_name}: decoded to zero bytes")

    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}: bytes decoded but could not be interpreted as an image"
        )

    return image


def decode_burst(frames, min_frames: int, max_frames: int):
    """
    frames: List[BurstFrame] (pydantic models with .image_b64 and .timestamp_ms)
    Returns: list of (np.ndarray BGR image, timestamp_ms, flash_state) sorted by timestamp.
    """
    if not frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    if len(frames) < min_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {min_frames} frames for reliable detection, got {len(frames)}"
        )

    if len(frames) > max_frames:
        frames = frames[:max_frames]

    decoded = []
    for i, f in enumerate(frames):
        img = decode_base64_image(f.image_b64, field_name=f"frames[{i}]")
        decoded.append((img, f.timestamp_ms, f.screen_flash_state))

    decoded.sort(key=lambda t: t[1])
    return decoded


def assert_consistent_resolution(images):
    """All frames in a burst must share resolution or downstream frame-diffing breaks silently."""
    if not images:
        return
    h0, w0 = images[0].shape[:2]
    for i, img in enumerate(images[1:], start=1):
        h, w = img.shape[:2]
        if (h, w) != (h0, w0):
            raise HTTPException(
                status_code=400,
                detail=f"Frame {i} resolution {w}x{h} does not match frame 0 resolution {w0}x{h0}"
            )
