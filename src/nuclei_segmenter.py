from typing import Dict, Any
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


# ============================================================
# ANNULUS KERNEL
# ============================================================
def _annulus_kernel(r_inner, r_outer):
    pad = int(np.ceil(r_outer * 1.5))
    y, x = np.mgrid[-pad:pad+1, -pad:pad+1]
    rr = np.sqrt(x**2 + y**2)

    band = ((rr >= r_inner) & (rr <= r_outer)).astype(np.float32)
    band = gaussian_filter(band, sigma=1.0)

    surround = np.exp(-0.5 * (rr / (1.2 * r_outer + 1e-8))**2)

    k = band - 0.15 * surround
    k -= k.mean()
    k /= np.sqrt((k * k).sum()) + 1e-8

    return k


# ============================================================
# NMS (TOP-K PEAKS)
# ============================================================
def _pick_top_k(resp, r_outer_map, r_inner_map, k=2):
    resp_copy = resp.copy()
    H, W = resp.shape
    yy, xx = np.mgrid[:H, :W]

    detections = []

    for _ in range(k):
        y, x = np.unravel_index(np.argmax(resp_copy), resp.shape)

        if not np.isfinite(resp_copy[y, x]):
            break

        r_out = int(r_outer_map[y, x])
        r_in = int(r_inner_map[y, x])

        detections.append((int(x), int(y), r_in, r_out))

        # suppress neighborhood
        suppress = (xx - x)**2 + (yy - y)**2 < (1.3 * r_out)**2
        resp_copy[suppress] = -np.inf

    return detections


# ============================================================
# MAIN SEGMENTATION
# ============================================================
def segment_nuclei(
    roi_stack: np.ndarray,
    channel_nucleus: int = 2,
    z_trim: int = 5,
) -> Dict[str, Any]:

    # ---------------------------------------------------
    # projection (simple + stable)
    # ---------------------------------------------------
    roi = roi_stack[z_trim:-z_trim]
    proj = roi.mean(axis=0).astype(np.float32)

    H, W = proj.shape
    yy, xx = np.mgrid[:H, :W]

    # ---------------------------------------------------
    # normalize
    # ---------------------------------------------------
    p1, p99 = np.percentile(proj, [1, 99])
    proj_n = np.clip((proj - p1) / (p99 - p1 + 1e-8), 0, 1)

    # ---------------------------------------------------
    # denoise
    # ---------------------------------------------------
    proj_s = gaussian_filter(proj_n, 0.8)

    # ---------------------------------------------------
    # bandpass (membrane enhancement)
    # ---------------------------------------------------
    high = gaussian_filter(proj_s, 1.5)
    low = gaussian_filter(proj_s, 7.0)
    band = high - low

    bg = np.percentile(band, 5)
    band = np.clip(band - bg, 0, None)
    band = band / (band.max() + 1e-8)

    # ---------------------------------------------------
    # radius search
    # ---------------------------------------------------
    r_outer_min = int(0.18 * min(H, W))
    r_outer_max = int(0.30 * min(H, W))
    r_outer_vals = np.arange(r_outer_min, r_outer_max + 1, 2)

    responses = []

    for r_out in r_outer_vals:
        r_in = int(0.70 * r_out)
        k = _annulus_kernel(r_in, r_out)

        resp = cv2.filter2D(band, -1, k, borderType=cv2.BORDER_REFLECT)
        responses.append(resp)

    resp_stack = np.stack(responses, axis=0)

    best_idx = np.argmax(resp_stack, axis=0)
    best_response = resp_stack.max(axis=0)

    best_outer = r_outer_vals[best_idx]
    best_inner = (0.65 * best_outer).astype(int)

    # ---------------------------------------------------
    # suppress borders (very important)
    # ---------------------------------------------------
    margin = int(0.9 * r_outer_max)
    best_response[:margin, :] = -np.inf
    best_response[-margin:, :] = -np.inf
    best_response[:, :margin] = -np.inf
    best_response[:, -margin:] = -np.inf

    # ---------------------------------------------------
    # detect top 2 nuclei
    # ---------------------------------------------------
    detections = _pick_top_k(best_response, best_outer, best_inner, k=2)

    if len(detections) == 0:
        raise RuntimeError("No nucleus detected.")

    # ---------------------------------------------------
    # build label map
    # ---------------------------------------------------
    roi_labels_2d = np.zeros((H, W), dtype=np.int32)

    centers = []
    radii = []

    for i, (x, y, r_in, r_out) in enumerate(detections, start=1):
        mask = (xx - x)**2 + (yy - y)**2 <= r_out**2

        roi_labels_2d[mask] = i

        centers.append([y, x])  # (y, x)
        radii.append(r_out)

    # ---------------------------------------------------
    # expand to 3D
    # ---------------------------------------------------
    
    
# ---------------------------------------------------
# build softer tapered 3D masks instead of harsh ellipsoid masks
# ---------------------------------------------------
    Z = roi_stack.shape[0]
    roi_labels_3d = np.zeros((Z, H, W), dtype=np.int32)

    z_thresh_frac = 0.25      # was 0.35; lower = include more top/bottom slices
    z_pad = 1                 # add extra z slices above/below valid range
    min_taper_frac = 0.65     # radius never shrinks below 65% inside valid z range
    mask_radius_scale = 1.08

    for i, ((cy, cx), r) in enumerate(zip(centers, radii), start=1):
        
        r_mask = r * mask_radius_scale
        base_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_mask**2

        z_profile = np.array([
            roi_stack[z][base_mask].mean() if np.any(base_mask) else 0.0
            for z in range(Z)
        ], dtype=np.float32)

        z_profile = gaussian_filter(z_profile, sigma=1.0)

        z_peak = int(np.argmax(z_profile))
        z_thresh = z_thresh_frac * z_profile[z_peak]

        valid_z = np.where(z_profile >= z_thresh)[0]

        if len(valid_z) == 0:
            z0, z1 = max(0, z_peak - 2), min(Z - 1, z_peak + 2)
        else:
            z0, z1 = int(valid_z.min()), int(valid_z.max())

        # expand z range slightly so top/bottom foci are not lost
        z0 = max(0, z0 - z_pad)
        z1 = min(Z - 1, z1 + z_pad)

        zc = 0.5 * (z0 + z1)
        rz = max((z1 - z0) / 2.0, 1.0)

        for z in range(Z):
            dz = (z - zc) / rz

            if abs(dz) > 1.0:
                continue

            # softer taper: do not let radius collapse at top/bottom
            taper = np.sqrt(max(0.0, 1.0 - dz**2))
            taper = max(min_taper_frac, taper)

            r_xy = r_mask * taper

            mask_z = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_xy**2
            roi_labels_3d[z, mask_z] = i

    roi_mask_3d = roi_labels_3d > 0


    return {
        "roi_proj": proj,
        "mask2d": roi_labels_2d > 0,
        "labels2d": roi_labels_2d,
        "roi_mask_3d": roi_mask_3d,
        "roi_labels_3d": roi_labels_3d,
        "centers": np.array(centers),
        "radii": radii,
    }
