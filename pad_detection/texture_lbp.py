"""
pad_detection/texture_lbp.py

Low-weight supporting signal (see rationale in config.py / README). Printed
paper and PVC ID cards have a fine, somewhat irregular microtexture from the
print/laminate process. An emissive digital display, even when its pixel
grid isn't independently resolvable (see moire_fft.py), still tends to
produce a more UNIFORM local texture statistic because it's a smooth glass
surface. LBP (Local Binary Patterns) uniformity is used as a coarse proxy.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern


def score_texture_lbp(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Downscale a bit for speed/noise robustness, LBP doesn't need full resolution
    gray_small = cv2.resize(gray, (400, 400))

    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_small, n_points, radius, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)

    # "Uniformity" of the LBP histogram: a very peaked histogram (texture
    # dominated by one or two pattern codes) suggests a smooth, glass-like
    # emissive surface. A flatter, more spread histogram suggests genuine
    # print/plastic microtexture variety.
    peak_mass = float(np.sort(hist)[-2:].sum())   # mass in the top-2 bins

    # Calibrate against your own captures.
    score = float(np.clip(1.0 - (peak_mass - 0.35) / 0.35, 0.0, 1.0))

    return {
        "score": score,
        "fired": True,
        "detail": f"LBP top-2-bin mass={peak_mass:.3f}",
    }
