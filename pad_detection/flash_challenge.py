"""
pad_detection/flash_challenge.py

Strongest PAD signal. The client briefly pulses its own display
(white -> off, or similar) during the capture burst while filming the
candidate's document. We measure whether the document region's brightness
correlates with that pulse.

- A physical object (paper/PVC card) reflects ambient light, so its
  brightness in-frame rises and falls with the pulse.
- A screen-in-frame (phone/monitor showing the doc) has its OWN backlight
  driving its brightness, largely independent of the room's light — so the
  correlation with our pulse is much weaker or absent.

This is PPI-independent, which is exactly why it doesn't share the failure
mode moire has on high-density phone displays.
"""

from typing import List, Tuple
import numpy as np
import cv2


def _mean_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def score_flash_challenge(frames_with_state: List[Tuple[np.ndarray, float, str]]) -> dict:
    """
    frames_with_state: list of (frame, timestamp_ms, screen_flash_state) sorted by time.
    Returns {"score": 0..1 (1=physical), "fired": bool, "detail": str}
    """
    # Need at least one "white" and one "off" labeled frame to correlate against.
    white_frames = [f for f, _, s in frames_with_state if s == "white"]
    off_frames = [f for f, _, s in frames_with_state if s == "off"]

    if len(white_frames) == 0 or len(off_frames) == 0:
        return {"score": 0.5, "fired": False,
                "detail": "No labeled flash-state frames in burst; signal not evaluated"}

    white_brightness = np.mean([_mean_brightness(f) for f in white_frames])
    off_brightness = np.mean([_mean_brightness(f) for f in off_frames])

    delta = white_brightness - off_brightness
    
    if delta < 0.0:
        # THE AE PANIC FIX: If the camera darkened the frame during the flash, 
        # a glossy physical card acted like a mirror and triggered auto-exposure compensation. 
        # Glowing screens absorb flash and stay near 0.0.
        score = 0.90
    else:
        # THE CLOSE-UP/TILT FIX: Screens max out around a 0.5 delta. 
        # If the overall frame is highly illuminated (close-up white card), 
        # the camera's AE actively dampens the delta. Scale down reference dynamically.
        if off_brightness > 110.0:
            PHYSICAL_DELTA_REFERENCE = 1.0  # Dynamic adjustment for close-up exposure dampening
        else:
           PHYSICAL_DELTA_REFERENCE = 6.0
        score = float(np.clip(delta / PHYSICAL_DELTA_REFERENCE, 0.0, 1.0))

    return {
        "score": score,
        "fired": True,
        "detail": f"brightness delta={delta:.2f} (white={white_brightness:.1f}, off={off_brightness:.1f})",
    }