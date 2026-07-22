from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
    median_filter,
    binary_erosion,
    label as ndi_label,
)

from src.utils import load_lsm, extract_channels
from src.nuclei_segmenter import segment_nuclei


# ============================================================
# OUTPUT CONTAINER
# ============================================================
@dataclass
class DetectionResult:
    file_path: str
    file_stem: str
    img: np.ndarray                  # shape: (Z, C, Y, X)
    roi_stack: np.ndarray            # nucleus channel, shape: (Z, Y, X)
    spot_stack: np.ndarray           # foci channel, shape: (Z, Y, X)

    roi_proj: np.ndarray
    mask2d: np.ndarray
    labels2d: np.ndarray

    roi_mask_3d: np.ndarray
    roi_labels_3d: np.ndarray

    blobs_raw: np.ndarray
    blobs_in_roi: np.ndarray
    blobs_filtered: np.ndarray

    nucleus1_blobs: np.ndarray
    nucleus2_blobs: np.ndarray

    nucleus1_count: int
    nucleus2_count: int

    threshold: float


# ============================================================
# ROBUST HELPERS
# ============================================================
def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-8)


def _safe_percentile(x: np.ndarray, q: float, default: float = 0.0) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    return float(np.percentile(x, q))


# ============================================================
# BACKGROUND SUBTRACTION + Z NORMALIZATION
# ============================================================
def _normalize_spot_stack(
    spot_stack: np.ndarray,
    roi_mask_3d: np.ndarray,
    bg_sigma_xy: float = 18.0,
    clip_percentiles: Tuple[float, float] = (0.1, 99.9),
) -> np.ndarray:
    """
    Convert raw foci channel to a background-subtracted, z-normalized stack.

    Why this is better than thresholding raw intensity directly:
    - removes broad nuclear/background haze
    - reduces slice-to-slice brightness drift / photobleaching effects
    - makes one threshold more comparable across z-slices
    """
    spot = spot_stack.astype(np.float32, copy=False)
    finite = spot[np.isfinite(spot)]
    if finite.size == 0:
        return np.zeros_like(spot, dtype=np.float32)

    lo, hi = np.percentile(finite, clip_percentiles)
    spot = np.clip(spot, lo, hi).astype(np.float32)

    # Estimate smooth background independently per z-slice.
    bg = gaussian_filter(spot, sigma=(0.0, bg_sigma_xy, bg_sigma_xy))
    sub = spot - bg
    sub[sub < 0] = 0.0

    norm = np.zeros_like(sub, dtype=np.float32)
    Z = sub.shape[0]
    for z in range(Z):
        mask_z = roi_mask_3d[z] if z < roi_mask_3d.shape[0] else np.zeros_like(sub[z], dtype=bool)
        vals = sub[z][mask_z]
        if vals.size < 32:
            vals = sub[z][np.isfinite(sub[z])]
        med = float(np.median(vals)) if vals.size else 0.0
        sig = _robust_sigma(vals)
        norm[z] = (sub[z] - med) / (sig + 1e-8)

    norm[~np.isfinite(norm)] = 0.0
    norm[norm < 0] = 0.0
    return norm.astype(np.float32)


# ============================================================
# MULTISCALE 3D DoG RESPONSE
# ============================================================
def _build_multiscale_3d_dog_response(
    spot_norm: np.ndarray,
    sigmas_xy: Tuple[float, ...] = (1.1, 1.4, 1.8, 2.2),
    dog_ratio: float = 2.4,
    sigma_z: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 3D-ish DoG response. The z smoothing is intentionally mild because
    microscope z-spacing is usually coarser than xy spacing.
    """
    responses = []
    radii = []

    for s in sigmas_xy:
        small = gaussian_filter(spot_norm, sigma=(sigma_z, s, s))
        large = gaussian_filter(
            spot_norm,
            sigma=(max(1.0, 2.0 * sigma_z), dog_ratio * s, dog_ratio * s),
        )
        resp = small - large
        resp[resp < 0] = 0.0

        # Scale normalization: prevents the smallest scale from always winning.
        resp *= float(s ** 1.2)
        responses.append(resp.astype(np.float32))
        radii.append(np.full_like(resp, 1.8 * s, dtype=np.float32))

    responses = np.stack(responses, axis=0)
    radii = np.stack(radii, axis=0)
    best_idx = np.argmax(responses, axis=0)
    best_response = np.max(responses, axis=0)
    best_radius = np.take_along_axis(radii, best_idx[None, ...], axis=0)[0]

    # Used for local prominence measurements, not thresholding.
    intensity_smooth = gaussian_filter(spot_norm, sigma=(0.4, 0.8, 0.8))
    return best_response.astype(np.float32), best_radius.astype(np.float32), intensity_smooth.astype(np.float32)


# ============================================================
# LOCAL 3D PROMINENCE FEATURES
# ============================================================
def _local_prominence_features_3d(
    intensity_3d: np.ndarray,
    z: int,
    y: int,
    x: int,
    radius_hint: float,
) -> Dict[str, float]:
    Z, H, W = intensity_3d.shape
    r_xy = max(1.5, float(radius_hint))
    patch_xy = max(5, int(np.ceil(3.5 * r_xy)))
    patch_z = 2

    z0 = max(0, z - patch_z)
    z1 = min(Z, z + patch_z + 1)
    y0 = max(0, y - patch_xy)
    y1 = min(H, y + patch_xy + 1)
    x0 = max(0, x - patch_xy)
    x1 = min(W, x + patch_xy + 1)

    patch = intensity_3d[z0:z1, y0:y1, x0:x1]
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]

    dz = (zz - z) / 1.5
    rr2 = (yy - y) ** 2 + (xx - x) ** 2

    center_r = max(1.2, 0.85 * r_xy)
    ring_in = max(2.2, 1.6 * r_xy)
    ring_out = max(ring_in + 1.0, 3.2 * r_xy)

    center_mask = (np.abs(dz) <= 1.0) & (rr2 <= center_r ** 2)
    ring_mask = (np.abs(dz) <= 1.5) & (rr2 >= ring_in ** 2) & (rr2 <= ring_out ** 2)

    center_vals = patch[center_mask]
    ring_vals = patch[ring_mask]

    if center_vals.size == 0 or ring_vals.size < 8:
        return {
            "center_signal": 0.0,
            "ring_median": 0.0,
            "prominence": 0.0,
            "ring_noise": 0.0,
            "snr": 0.0,
        }

    # Percentile is less noisy than max and less diluted than mean.
    center_signal = float(np.percentile(center_vals, 85))
    ring_median = float(np.median(ring_vals))
    ring_noise = _robust_sigma(ring_vals)
    prominence = center_signal - ring_median
    snr = prominence / (ring_noise + 1e-8)

    return {
        "center_signal": center_signal,
        "ring_median": ring_median,
        "prominence": float(prominence),
        "ring_noise": float(ring_noise),
        "snr": float(snr),
    }


# ============================================================
# MASK HANDLING / ASSIGNMENT
# ============================================================
def _erode_labels_3d(roi_labels_3d: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Erode each nucleus label independently so boundary noise does not create
    confident but wrong assignments. If erosion removes a tiny label, fall back
    to the original label for that nucleus.
    """
    if iterations <= 0:
        return roi_labels_3d

    out = np.zeros_like(roi_labels_3d, dtype=np.int32)
    structure = np.ones((1, 3, 3), dtype=bool)  # xy-only erosion

    for label_id in [1, 2]:
        mask = roi_labels_3d == label_id
        if not np.any(mask):
            continue
        eroded = binary_erosion(mask, structure=structure, iterations=iterations)
        if np.count_nonzero(eroded) < 0.25 * np.count_nonzero(mask):
            eroded = mask
        out[eroded] = label_id
    return out


def _assign_label_by_local_majority(
    roi_labels_3d: np.ndarray,
    z: int,
    y: int,
    x: int,
    radius_hint: float,
) -> int:
    """
    Assign a spot to the nucleus occupying most pixels around the spot center,
    rather than trusting one boundary voxel.
    """
    Z, H, W = roi_labels_3d.shape
    rz = 1
    rxy = max(2, int(np.ceil(0.8 * radius_hint)))

    z0 = max(0, z - rz)
    z1 = min(Z, z + rz + 1)
    y0 = max(0, y - rxy)
    y1 = min(H, y + rxy + 1)
    x0 = max(0, x - rxy)
    x1 = min(W, x + rxy + 1)

    patch = roi_labels_3d[z0:z1, y0:y1, x0:x1]
    labels, counts = np.unique(patch[patch > 0], return_counts=True)
    if labels.size == 0:
        return 0
    return int(labels[np.argmax(counts)])


# ============================================================
# 3D CANDIDATE EXTRACTION + MERGE
# ============================================================
def _merge_close_candidates(
    candidates: List[dict],
    merge_radius_xy: float = 4.5,
    merge_radius_z: int = 1,
) -> List[dict]:
    """
    Greedy final merge. This catches split maxima from one physical focus.
    It does not merge across nucleus labels.
    """
    kept: List[dict] = []
    for cand in sorted(candidates, key=lambda d: d["score"], reverse=True):
        duplicate = False
        for prev in kept:
            if cand["label_id"] != prev["label_id"]:
                continue
            if abs(cand["z"] - prev["z"]) > merge_radius_z:
                continue
            if np.hypot(cand["y"] - prev["y"], cand["x"] - prev["x"]) <= merge_radius_xy:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return sorted(kept, key=lambda d: (d["label_id"], d["z"], d["y"], d["x"]))


def _extract_3d_candidates(
    response_3d: np.ndarray,
    intensity_3d: np.ndarray,
    radius_map: np.ndarray,
    roi_labels_3d: np.ndarray,
    peak_mad_k: float = 5.0,
    peak_percentile: float | None = 97.5,
    min_distance_xy: int = 6,
    min_distance_z: int = 1,
    min_radius_px: float = 2.0,
    max_radius_px: float = 6.5,
    prominence_noise_k: float = 3.0,
    min_prominence_abs: float = 1.2,
    boundary_erosion_px: int = 1,
) -> Tuple[List[dict], float]:
    roi_core = _erode_labels_3d(roi_labels_3d, iterations=boundary_erosion_px) > 0
    vals = response_3d[roi_core]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return [], float("nan")

    med = float(np.median(vals))
    sig = _robust_sigma(vals)
    threshold = med + peak_mad_k * sig

    # A mild percentile floor prevents very noisy stacks from exploding, but this
    # is deliberately much lower than 99.5 so dense/high-count nuclei are not crushed.
    if peak_percentile is not None:
        threshold = max(threshold, _safe_percentile(vals, peak_percentile, threshold))

    local_max = response_3d == maximum_filter(
        response_3d,
        size=(2 * min_distance_z + 1, 2 * min_distance_xy + 1, 2 * min_distance_xy + 1),
        mode="nearest",
    )
    cand_mask = local_max & (response_3d >= threshold) & roi_core

    labeled, ncomp = ndi_label(cand_mask)
    candidates: List[dict] = []

    for comp_id in range(1, ncomp + 1):
        zs, ys, xs = np.where(labeled == comp_id)
        if zs.size == 0:
            continue

        scores = response_3d[zs, ys, xs]
        best_i = int(np.argmax(scores))
        z = int(zs[best_i])
        y = int(ys[best_i])
        x = int(xs[best_i])
        r = float(radius_map[z, y, x])

        if r < min_radius_px or r > max_radius_px:
            continue

        feats = _local_prominence_features_3d(intensity_3d, z=z, y=y, x=x, radius_hint=r)
        if feats["prominence"] < min_prominence_abs:
            continue
        if feats["snr"] < prominence_noise_k:
            continue

        label_id = _assign_label_by_local_majority(roi_labels_3d, z=z, y=y, x=x, radius_hint=r)
        if label_id not in (1, 2):
            continue

        candidates.append({
            "z": z,
            "y": y,
            "x": x,
            "radius": r,
            "score": float(response_3d[z, y, x]),
            "prominence": feats["prominence"],
            "snr": feats["snr"],
            "label_id": label_id,
        })

    candidates = _merge_close_candidates(
        candidates,
        merge_radius_xy=max(3.0, 0.75 * min_distance_xy),
        merge_radius_z=min_distance_z,
    )
    return candidates, float(threshold)


# ============================================================
# MAIN SPOT DETECTOR
# ============================================================
def detect_spots(
    spot_stack: np.ndarray,
    roi_mask_3d: np.ndarray,
    roi_labels_3d: np.ndarray,
    # spot size
    sigmas_xy: Tuple[float, ...] = (1.6, 2.2, 2.6), #what size blobs to enhance
    dog_ratio: float = 2.0, #how much local background to subtract
    sigma_z: float = 0.7,
    min_radius_px: float = 3.0,
    max_radius_px: float = 6.0,
    # intensity / response threshold
    peak_percentile: float | None = 99.5,
    peak_mad_k: float = 11.0,
    # local prominence filter
    prominence_noise_k: float = 8.0,
    min_prominence_abs: float = 6.0,
    # peak separation
    min_distance_xy: int = 10,
    min_distance_z: int = 3,
    # kept for backward compatibility with old tuning scripts
    link_radius_xy: float | None = None,
    max_gap_z: int | None = None,
    min_track_length: int | None = None,
    # preprocessing / mask handling
    bg_sigma_xy: float = 18.0,
    boundary_erosion_px: int = 1,
) -> Dict[str, Any]:
    """
    Robust foci detector.

    Main change from the old code:
    - detect 3D local maxima directly instead of slice peaks + greedy z-linking
    - normalize background and z-slice intensity before thresholding
    - use mild percentile floor + MAD threshold, not a hard 99.5 percentile gate
    - assign each focus by local majority label around the spot center
    """
    if not np.any(roi_mask_3d):
        raise RuntimeError("ROI mask is empty.")

    spot_norm = _normalize_spot_stack(
    spot_stack=spot_stack,
    roi_mask_3d=roi_mask_3d,
    bg_sigma_xy=bg_sigma_xy,
    )

    # remove fine grain / salt-pepper noise before DoG amplifies it
    spot_norm = median_filter(spot_norm, size=(1, 5, 5))

    response_img, radius_map, intensity_img = _build_multiscale_3d_dog_response(
        spot_norm=spot_norm,
        sigmas_xy=sigmas_xy,
        dog_ratio=dog_ratio,
        sigma_z=sigma_z,
    )

    response_img = response_img.copy()
    response_img[~roi_mask_3d] = 0.0

    candidates, threshold = _extract_3d_candidates(
        response_3d=response_img,
        intensity_3d=intensity_img,
        radius_map=radius_map,
        roi_labels_3d=roi_labels_3d,
        peak_mad_k=peak_mad_k,
        peak_percentile=peak_percentile,
        min_distance_xy=min_distance_xy,
        min_distance_z=min_distance_z,
        min_radius_px=min_radius_px,
        max_radius_px=max_radius_px,
        prominence_noise_k=prominence_noise_k,
        min_prominence_abs=min_prominence_abs,
        boundary_erosion_px=boundary_erosion_px,
    )

    def to_array(label_id: int) -> np.ndarray:
        pts = [c for c in candidates if c["label_id"] == label_id]
        if not pts:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(
            [[c["z"], c["y"], c["x"], c["radius"]] for c in pts],
            dtype=np.float32,
        )

    nucleus1_blobs = to_array(1)
    nucleus2_blobs = to_array(2)
    blobs_filtered = (
        np.vstack([a for a in [nucleus1_blobs, nucleus2_blobs] if len(a) > 0]).astype(np.float32)
        if (len(nucleus1_blobs) + len(nucleus2_blobs)) > 0
        else np.empty((0, 4), dtype=np.float32)
    )

    # Here raw == filtered because extraction already performs biologically meaningful filters.
    # Kept as separate keys for UI/backward compatibility.
    return {
        "threshold": float(threshold) if np.isfinite(threshold) else 0.0,
        "blobs_raw": blobs_filtered,
        "blobs_in_roi": blobs_filtered,
        "blobs_filtered": blobs_filtered,
        "nucleus1_blobs": nucleus1_blobs,
        "nucleus2_blobs": nucleus2_blobs,
        "nucleus1_count": int(len(nucleus1_blobs)),
        "nucleus2_count": int(len(nucleus2_blobs)),
    }


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================
def run_detection(
    file_path: str,
    roi_channel: int = 2,
    spot_channel: int = 1,
    spot_params: dict | None = None,
) -> DetectionResult:
    if spot_params is None:
        spot_params = {}

    img = load_lsm(file_path)
    roi_stack, spot_stack = extract_channels(
        img,
        roi_channel=roi_channel,
        spot_channel=spot_channel,
    )

    # Trim noisy top/bottom slices once here.
    roi_stack = roi_stack[5:-5]
    spot_stack = spot_stack[5:-5]

    seg = segment_nuclei(roi_stack)

    det = detect_spots(
        spot_stack=spot_stack,
        roi_mask_3d=seg["roi_mask_3d"],
        roi_labels_3d=seg["roi_labels_3d"],
        **spot_params,
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


# ============================================================
# TUNING HELPERS
# ============================================================
def prepare_detection_inputs(
    file_path: str,
    roi_channel: int = 2,
    spot_channel: int = 1,
):
    img = load_lsm(file_path)
    roi_stack, spot_stack = extract_channels(
        img,
        roi_channel=roi_channel,
        spot_channel=spot_channel,
    )

    roi_stack = roi_stack[5:-5]
    spot_stack = spot_stack[5:-5]
    seg = segment_nuclei(roi_stack)

    return {
        "file_path": str(file_path),
        "file_stem": Path(file_path).stem,
        "img": img,
        "roi_stack": roi_stack,
        "spot_stack": spot_stack,
        "roi_mask_3d": seg["roi_mask_3d"],
        "roi_labels_3d": seg["roi_labels_3d"],
    }


def detect_from_prepared(prepared, spot_params=None):
    if spot_params is None:
        spot_params = {}

    det = detect_spots(
        spot_stack=prepared["spot_stack"],
        roi_mask_3d=prepared["roi_mask_3d"],
        roi_labels_3d=prepared["roi_labels_3d"],
        **spot_params,
    )

    return {
        "nucleus1_count": det["nucleus1_count"],
        "nucleus2_count": det["nucleus2_count"],
        "nucleus1_blobs": det["nucleus1_blobs"],
        "nucleus2_blobs": det["nucleus2_blobs"],
        "blobs_raw": det["blobs_raw"],
        "blobs_in_roi": det["blobs_in_roi"],
        "blobs_filtered": det["blobs_filtered"],
        "threshold": det["threshold"],
    }
