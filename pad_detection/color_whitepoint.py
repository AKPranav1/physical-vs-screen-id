"""
pad_detection/color_whitepoint.py

Digital displays tend to run cooler and more saturated than a physical
document reflecting typical indoor/warm ambient light. We estimate a rough
white-point/color-temperature proxy from the document region's brightest
neutral-ish pixels and check saturation statistics.
"""

import numpy as np
import cv2


def score_color_whitepoint(frame: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(frame.astype(np.float32))

    # Sample the brightest 5% of pixels as a proxy for the document's "white" reference area
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = np.percentile(gray, 95)
    mask = gray >= thresh

    if mask.sum() < 20:
        return {"score": 0.5, "fired": False, "detail": "Not enough bright reference pixels"}

    mean_b, mean_g, mean_r = b[mask].mean(), g[mask].mean(), r[mask].mean()

    # Blue/red ratio as a coarse color-temperature proxy: notably blue-shifted
    # whites (ratio well above 1) are more typical of an emissive LCD/OLED panel;
    # values closer to 1 (slightly warm, as under indoor tungsten/LED room light)
    # are more typical of a physical surface.
    blue_red_ratio = mean_b / max(1.0, mean_r)

    mean_saturation = hsv[:, :, 1][mask].mean() / 255.0

    # Calibrate these reference points against your own captures.
    COOL_RATIO_REFERENCE = 1.25   # ratio at/above this looks screen-like
    WARM_RATIO_REFERENCE = 0.95   # ratio at/below this looks physical-like

    if blue_red_ratio <= WARM_RATIO_REFERENCE:
        color_score = 1.0
    elif blue_red_ratio >= COOL_RATIO_REFERENCE:
        color_score = 0.0
    else:
        color_score = float(1.0 - (blue_red_ratio - WARM_RATIO_REFERENCE) /
                             (COOL_RATIO_REFERENCE - WARM_RATIO_REFERENCE))

    # Very high saturation in the "white" reference area is also a screen tell
    saturation_penalty = float(np.clip(mean_saturation * 2.0, 0.0, 0.3))
    score = float(np.clip(color_score - saturation_penalty, 0.0, 1.0))

    return {
        "score": score,
        "fired": True,
        "detail": f"blue/red ratio={blue_red_ratio:.3f}, mean saturation={mean_saturation:.3f}",
    }
