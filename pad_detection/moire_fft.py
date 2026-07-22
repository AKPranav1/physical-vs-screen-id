"""
pad_detection/moire_fft.py

Bonus signal, intentionally LOW WEIGHT (see config.PAD_SIGNAL_WEIGHTS and the
README section "Why moire is weighted low"). Modern high-PPI phone screens
photographed at normal document-verification distance by a typical webcam
frequently do NOT alias into visible moire — the fine pixel grid is below
the webcam's resolvable frequency and gets smoothly downsampled instead.

When it DOES fire (older/lower-PPI displays, closer distance, some Android
panels with less display-side anti-aliasing filtering), it's a strong
positive, so we keep it in the fusion — just not relied upon.

Averaging the FFT magnitude spectrum across several burst frames reduces
noise vs. a single-frame check.
"""

import numpy as np
import cv2


def _fft_magnitude(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    return np.log(np.abs(fshift) + 1.0)


def score_moire_fft(frames: list) -> dict:
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # Resize all to a common size for consistent averaging
    target_shape = (512, 512)
    resized = [cv2.resize(g, target_shape) for g in grays]

    spectra = [_fft_magnitude(g) for g in resized]
    avg_spectrum = np.mean(spectra, axis=0)

    h, w = avg_spectrum.shape
    cy, cx = h // 2, w // 2

    # Exclude the low-frequency DC-heavy center (document's own shape/edges)
    # and look at the mid-frequency ring where periodic pixel-grid artifacts
    # would show up as a moire beat pattern.
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    ring_mask = (dist > 0.08 * h) & (dist < 0.35 * h)

    ring_values = avg_spectrum[ring_mask]
    ring_mean = ring_values.mean()
    ring_std = ring_values.std()

    # Sharp, isolated peaks well above the ring's own local statistics indicate
    # a periodic interference pattern (moire). A flat ring (as expected on a
    # smoothly-downsampled high-PPI screen or genuine physical document) does not.
    peak_threshold = ring_mean + 3.5 * ring_std
    peak_ratio = float((ring_values > peak_threshold).mean())

    # More peak energy -> more likely moire -> more likely screen recapture
    score = float(np.clip(1.0 - peak_ratio * 20.0, 0.0, 1.0))

    fired = peak_ratio > 0.0005  # only "fires" (counts strongly) if it actually found something

    return {
        "score": score,
        "fired": fired,
        "detail": f"moire peak ratio={peak_ratio:.5f} (low-weight signal by design)",
    }
