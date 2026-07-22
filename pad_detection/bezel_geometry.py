"""
pad_detection/bezel_geometry.py

Detects a device bezel (phone/monitor edge) around the document region.
A screen recapture very often shows a second, larger rectangular boundary
(the device's physical edge) concentric with or surrounding the document
image on that screen — a physical card/paper doesn't have this second frame.

Approach: Hough line detection on the region surrounding the document's own
detected boundary; look for a second set of long, straight, roughly-parallel
lines outside the document contour. Purely geometric, PPI-independent.
"""

import numpy as np
import cv2


def _find_document_contour(gray: np.ndarray):
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * gray.shape[0] * gray.shape[1]:
        return None
    return largest


def score_bezel_geometry(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    doc_contour = _find_document_contour(gray)
    if doc_contour is None:
        return {"score": 0.5, "fired": False, "detail": "Could not localize document contour"}

    x, y, cw, ch = cv2.boundingRect(doc_contour)
    
    # THE EDGE-TO-EDGE FIX
    # Calculate how much of the camera frame the document is taking up.
    doc_area_ratio = (cw * ch) / (h * w)
    
    if doc_area_ratio > 0.80:
        # If the document is taking up more than 80% of the screen, the user is 
        # shoving it directly into the lens. This is a classic "bezel hiding" attack.
        # We heavily penalize this instead of giving it a free pass.
        return {
            "score": 0.15,
            "fired": True,
            "detail": f"Document fills {doc_area_ratio*100:.1f}% of frame (suspicious edge-to-edge presentation)"
        }

    # Look at a margin band OUTSIDE the document's own bounding box for a second,
    # larger rectangular edge (the bezel). Expand outward by 25% of doc size.
    margin_x, margin_y = int(cw * 0.25), int(ch * 0.25)
    x0, y0 = max(0, x - margin_x), max(0, y - margin_y)
    x1, y1 = min(w, x + cw + margin_x), min(h, y + ch + margin_y)

    band = gray[y0:y1, x0:x1]
    edges = cv2.Canny(band, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=int(0.5 * min(band.shape)), maxLineGap=10)

    if lines is None or len(lines) == 0:
        return {"score": 0.8, "fired": True,
                "detail": "No secondary bezel-like edges detected around document"}

    # Count long lines roughly parallel to the document's own edges (0 or 90 degrees +/- 10deg)
    bezel_like = 0
    for line in lines:
        x1_, y1_, x2_, y2_ = line[0]
        angle = np.degrees(np.arctan2(y2_ - y1_, x2_ - x1_)) % 180
        if angle < 10 or angle > 170 or (80 < angle < 100):
            bezel_like += 1

    # More bezel-like straight edges outside the doc contour -> more likely a
    # device frame is present -> more likely a screen recapture.
    density = bezel_like / max(1, len(lines))
    score = float(np.clip(1.0 - density, 0.0, 1.0))

    return {
        "score": score,
        "fired": True,
        "detail": f"{bezel_like}/{len(lines)} candidate bezel edges detected outside document contour",
    }