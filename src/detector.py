from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy.spatial import cKDTree
from skimage.filters import gaussian
from skimage.feature import blob_log

from utils import load_lsm, extract_channels
from nuclei_segmenter import segment_nuclei



@dataclass
class DetectionResult:
    file_path: str
    file_stem: str
    img: np.ndarray                  # shape: (Z, C, Y, X)
    roi_stack: np.ndarray            # nucleus channel, shape: (Z, Y, X)
    spot_stack: np.ndarray           # foci channel, shape: (Z, Y, X)

    roi_proj: np.ndarray             # max projection of roi_stack
    mask2d: np.ndarray               # 2D binary nucleus mask
    labels2d: np.ndarray             # 2D nucleus labels (1, 2)

    roi_mask_3d: np.ndarray          # repeated 3D boolean mask
    roi_labels_3d: np.ndarray        # repeated 3D nucleus labels

    blobs_raw: np.ndarray            # raw blob_log output
    blobs_in_roi: np.ndarray         # blobs whose centers lie inside ROI
    blobs_filtered: np.ndarray       # after duplicate merge

    nucleus1_blobs: np.ndarray
    nucleus2_blobs: np.ndarray

    nucleus1_count: int
    nucleus2_count: int

    threshold: float




def detect_spots(
    spot_stack: np.ndarray,
    roi_mask_3d: np.ndarray,
    roi_labels_3d: np.ndarray,
    gaussian_sigma_3d: tuple[float, float, float] = (1, 1, 1),
    threshold_percentile: float = 99.5,
    min_sigma: float = 1.8,
    max_sigma: float = 4.0,
    num_sigma: int = 8,
    merge_radius: float = 3.0,
) -> Dict[str, Any]:
    """
    Detect spots only inside the segmented nuclei.
    """

    # smooth
    spot_smooth = gaussian(spot_stack.astype(np.float32), sigma=gaussian_sigma_3d)

    # restrict to nuclei only
    spot_masked = spot_smooth.copy()
    spot_masked[~roi_mask_3d] = 0

    vals = spot_masked[roi_mask_3d]
    if vals.size == 0:
        raise RuntimeError("ROI mask is empty. Cannot compute detection threshold.")

    threshold = float(np.percentile(vals, threshold_percentile))

    # blob detection
    blobs = blob_log(
        spot_masked,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )

    if blobs is None or len(blobs) == 0:
        empty = np.empty((0, 4), dtype=np.float32)
        return {
            "threshold": threshold,
            "blobs_raw": empty,
            "blobs_in_roi": empty,
            "blobs_filtered": empty,
            "nucleus1_blobs": empty,
            "nucleus2_blobs": empty,
            "nucleus1_count": 0,
            "nucleus2_count": 0,
        }

    blobs = np.asarray(blobs, dtype=np.float32)

    # keep only blobs whose centers fall inside ROI
    blobs_in_roi = []
    zmax, ymax, xmax = roi_mask_3d.shape

    for blob in blobs:
        z, y, x, r = blob
        zi = int(round(z))
        yi = int(round(y))
        xi = int(round(x))

        if (
            0 <= zi < zmax
            and 0 <= yi < ymax
            and 0 <= xi < xmax
            and roi_mask_3d[zi, yi, xi]
        ):
            blobs_in_roi.append(blob)

    blobs_in_roi = (
        np.asarray(blobs_in_roi, dtype=np.float32)
        if len(blobs_in_roi) > 0
        else np.empty((0, 4), dtype=np.float32)
    )

    # merge duplicates
    if len(blobs_in_roi) > 0:
        coords = blobs_in_roi[:, :3]
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=merge_radius)

        to_remove = set()
        for i, j in pairs:
            to_remove.add(j)

        keep_idx = [i for i in range(len(blobs_in_roi)) if i not in to_remove]
        blobs_filtered = blobs_in_roi[keep_idx]
    else:
        blobs_filtered = np.empty((0, 4), dtype=np.float32)

    # assign to nuclei
    nucleus1_blobs = []
    nucleus2_blobs = []

    for blob in blobs_filtered:
        z, y, x, r = blob
        zi = int(round(z))
        yi = int(round(y))
        xi = int(round(x))

        label = roi_labels_3d[zi, yi, xi]

        if label == 1:
            nucleus1_blobs.append(blob)
        elif label == 2:
            nucleus2_blobs.append(blob)

    nucleus1_blobs = (
        np.asarray(nucleus1_blobs, dtype=np.float32)
        if len(nucleus1_blobs) > 0
        else np.empty((0, 4), dtype=np.float32)
    )
    nucleus2_blobs = (
        np.asarray(nucleus2_blobs, dtype=np.float32)
        if len(nucleus2_blobs) > 0
        else np.empty((0, 4), dtype=np.float32)
    )

    return {
        "threshold": threshold,
        "blobs_raw": blobs,
        "blobs_in_roi": blobs_in_roi,
        "blobs_filtered": blobs_filtered,
        "nucleus1_blobs": nucleus1_blobs,
        "nucleus2_blobs": nucleus2_blobs,
        "nucleus1_count": int(len(nucleus1_blobs)),
        "nucleus2_count": int(len(nucleus2_blobs)),
    }





def run_detection(
    file_path: str,
    roi_channel: int = 0,
    spot_channel: int = 1,
) -> DetectionResult:
    """
    Main black-box function.
    This is the one the UI will call.
    """

    img = load_lsm(file_path)
    roi_stack, spot_stack = extract_channels(
        img,
        roi_channel=roi_channel,
        spot_channel=spot_channel,
    )

    seg = segment_nuclei(roi_stack)
    det = detect_spots(
        spot_stack=spot_stack,
        roi_mask_3d=seg["roi_mask_3d"],
        roi_labels_3d=seg["roi_labels_3d"],
    )

    return DetectionResult(
        file_path=str(file_path),
        file_stem=Path(file_path).stem,
        img=img,
        roi_stack=roi_stack,
        spot_stack=spot_stack,

        roi_proj=seg["roi_proj"],
        mask2d=seg["mask2d"],
        labels2d=seg["labels2d"],

        roi_mask_3d=seg["roi_mask_3d"],
        roi_labels_3d=seg["roi_labels_3d"],

        blobs_raw=det["blobs_raw"],
        blobs_in_roi=det["blobs_in_roi"],
        blobs_filtered=det["blobs_filtered"],

        nucleus1_blobs=det["nucleus1_blobs"],
        nucleus2_blobs=det["nucleus2_blobs"],

        nucleus1_count=det["nucleus1_count"],
        nucleus2_count=det["nucleus2_count"],

        threshold=det["threshold"],
    )