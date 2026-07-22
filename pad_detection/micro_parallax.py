"""
pad_detection/micro_parallax.py

Under the natural hand tremor of someone holding a document up to a webcam,
a REAL card/paper has physical relief (embossing, slight curvature, edge
thickness) that produces small differential motion across its surface as
the viewing angle micro-shifts. A flat photo (whether printed or shown on a
screen) is a single rigid plane, so its surface points all move together
uniformly (or the whole plane is static because it's clamped to a stand).

Approach: track a sparse set of feature points across the burst with
Lucas-Kanade optical flow, then measure how well a single planar homography
explains all the tracked point motion. Higher residual (non-planar motion) is
consistent with a genuine 3D object; a very good planar fit is consistent
with a flat recapture.

This is a supporting signal (low-medium weight) — a very steady hand or a
propped-up phone can suppress this effect, so it should never be decisive
alone.
"""

import numpy as np
import cv2


def score_micro_parallax(frames: list) -> dict:
    if len(frames) < 5:
        return {"score": 0.5, "fired": False, "detail": "Not enough frames for optical flow analysis"}

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    feature_params = dict(maxCorners=150, qualityLevel=0.15, minDistance=15, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p0 = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **feature_params)
    if p0 is None or len(p0) < 12:
        return {"score": 0.5, "fired": False, "detail": "Not enough trackable features on document surface"}

    residuals = []
    prev_gray, prev_pts = gray_frames[0], p0

    for gray in gray_frames[1:]:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        if next_pts is None:
            continue

        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        if len(good_prev) < 8:
            prev_gray, prev_pts = gray, next_pts if next_pts is not None else prev_pts
            continue

        H, inlier_mask = cv2.findHomography(good_prev, good_next, cv2.RANSAC, 3.0)
        if H is None:
            prev_gray, prev_pts = gray, next_pts
            continue

        projected = cv2.perspectiveTransform(good_prev.reshape(-1, 1, 2), H)
        residual = np.linalg.norm(projected.reshape(-1, 2) - good_next.reshape(-1, 2), axis=1)
        residuals.append(float(residual.mean()))

        prev_gray, prev_pts = gray, good_next.reshape(-1, 1, 2)

    if not residuals:
        return {"score": 0.5, "fired": False, "detail": "Optical flow tracking failed across burst"}

    mean_residual = float(np.mean(residuals))

    # Calibrate against your own captures: a rigid flat plane fits homography
    # near-perfectly (residual close to 0px); genuine 3D relief + natural hand
    # shake produces a small but consistently non-zero residual.
    RESIDUAL_REFERENCE_PX = 0.6
    score = float(np.clip(mean_residual / RESIDUAL_REFERENCE_PX, 0.0, 1.0))

    return {
        "score": score,
        "fired": True,
        "detail": f"mean homography residual={mean_residual:.3f}px over {len(residuals)} frame-pairs",
    }
