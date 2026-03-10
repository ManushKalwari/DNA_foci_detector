from typing import Dict, Any
import numpy as np

from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    disk,
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi



def segment_nuclei(
    roi_stack: np.ndarray,
    threshold_scale: float = 0.75,
    gaussian_sigma: float = 2.0,
    min_object_size: int = 2000,
    opening_radius: int = 4,
    closing_radius: int = 6,
    peak_min_distance: int = 60,
    expected_nuclei: int = 2,
) -> Dict[str, Any]:
    """
    2D max-projection based nucleus segmentation.
    Broadcasts the 2D result across Z, matching your current validated workflow.
    """

    # max projection
    roi_proj = np.max(roi_stack, axis=0).astype(np.float32)

    # smooth projection
    roi_proj_s = gaussian(roi_proj, sigma=gaussian_sigma)

    # threshold
    t = threshold_otsu(roi_proj_s)
    mask2d = roi_proj_s > (t * threshold_scale)

    # morphology cleanup
    mask2d = remove_small_objects(mask2d, min_object_size)
    mask2d = ndi.binary_fill_holes(mask2d)
    mask2d = binary_opening(mask2d, disk(opening_radius))
    mask2d = binary_closing(mask2d, disk(closing_radius))

    # distance transform
    distance = ndi.distance_transform_edt(mask2d)

    # find nucleus centers
    coords = peak_local_max(
        distance,
        labels=mask2d,
        min_distance=peak_min_distance,
        num_peaks=expected_nuclei,
    )

    # handle cases where only one nucleus exists
    if len(coords) == 0:
        raise RuntimeError("No nuclei detected.")

    # adapt automatically
    num_nuclei = len(coords)

    markers = np.zeros_like(mask2d, dtype=np.int32)

    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    markers = ndi.label(markers)[0]

    # watershed split
    labels2d = watershed(-distance, markers, mask=mask2d)

    # expand to 3D
    roi_mask_3d = np.repeat(mask2d[None, :, :], roi_stack.shape[0], axis=0).astype(bool)
    roi_labels_3d = np.repeat(labels2d[None, :, :], roi_stack.shape[0], axis=0).astype(np.int32)

    return {
        "roi_proj": roi_proj,
        "mask2d": mask2d,
        "labels2d": labels2d,
        "roi_mask_3d": roi_mask_3d,
        "roi_labels_3d": roi_labels_3d,
    }