"""
pad_detection/specular_reflection.py

Glass screens under indoor lighting produce small, sharp, high-contrast
specular highlights (mirror-like glare spots). Paper and PVC ID cards produce
broader, softer, lower-contrast diffuse highlights. This is independent of
screen PPI, which is why it stays useful where moire fails.

Approach: find very-bright, small, high-contrast connected components
(candidate glare spots) and characterize their size/sharpness. Many small,
extremely bright, sharp-edged spots -> glass-like -> more likely a screen.
"""

import numpy as np
import cv2


def score_specular_reflection(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Candidate glare = very bright pixels
    _, bright_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

    if num_labels <= 1:
        return {"score": 0.6, "fired": True, "detail": "No strong glare spots detected (mildly favors physical)"}

    doc_area = gray.shape[0] * gray.shape[1]
    sharp_small_spots = 0
    total_spots = 0

    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4 or area > 0.02 * doc_area:
            continue   # ignore noise pixels and huge bright regions (e.g. blown-out background)

        total_spots += 1
        x, y, cw, ch, _ = stats[i]
        pad = 3
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray.shape[1], x + cw + pad), min(gray.shape[0], y + ch + pad)
        patch = gray[y0:y1, x0:x1].astype(np.float32)

        if patch.size == 0:
            continue

        # sharp edge -> high local gradient variance around a SMALL bright blob
        grad = cv2.Laplacian(patch, cv2.CV_32F).var()
        if grad > 800 and area < 0.002 * doc_area:
            sharp_small_spots += 1

    if total_spots == 0:
        return {"score": 0.6, "fired": True, "detail": "No qualifying glare spots after filtering"}

    sharp_ratio = sharp_small_spots / total_spots
    # High ratio of small+sharp glare spots -> glass-like -> favors "screen"
    score = float(np.clip(1.0 - sharp_ratio, 0.0, 1.0))

    return {
        "score": score,
        "fired": True,
        "detail": f"{sharp_small_spots}/{total_spots} glare spots were small+sharp (glass-like)",
    }
