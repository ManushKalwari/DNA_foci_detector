import tifffile as tiff
import numpy as np
from pathlib import Path
import pandas as pd



def load_lsm(file_path: str) -> np.ndarray:
    """
    Load .lsm / tif stack.
    Expected shape from your notebook: (Z, C, Y, X)
    """
    with tiff.TiffFile(file_path) as tif:
        img = tif.asarray()

    if img.ndim != 4:
        raise ValueError(
            f"Expected image shape (Z, C, Y, X), got {img.shape}"
        )

    return img



def extract_channels(
    img: np.ndarray,
    roi_channel: int = 2,
    spot_channel: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    roi_channel: channel used for nucleus segmentation
    spot_channel: channel used for foci detection
    """
    if img.shape[1] <= max(roi_channel, spot_channel):
        raise ValueError(
            f"Channel index out of range. Image shape is {img.shape}"
        )

    roi_stack = img[:, roi_channel].astype(np.float32)
    spot_stack = img[:, spot_channel].astype(np.float32)

    return roi_stack, spot_stack





def save_counts_csv(results, output_dir: str | None = None) -> str:
    """
    Save combined CSV for multiple images.

    Output format:
    filename_cellnumber, spot_count

    B Image 1_cell1, 10
    B Image 1_cell2, 0
    C Image 3_cell1, 12
    C Image 3_cell2, 5
    """

    if not results:
        raise ValueError("No detection results to save.")

    # determine save location
    first_file = Path(results[0].file_path)

    out_dir = Path(output_dir) if output_dir is not None else first_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for r in results:

        rows.append({
            "filename_cellnumber": f"{r.file_stem}_nucleus1",
            "spot_count": int(r.nucleus1_count),
        })

        rows.append({
            "filename_cellnumber": f"{r.file_stem}_nucleus2",
            "spot_count": int(r.nucleus2_count),
        })

    df = pd.DataFrame(rows)

    csv_path = out_dir / "spot_counts_all_images.csv"

    df.to_csv(csv_path, index=False)

    return str(csv_path)
