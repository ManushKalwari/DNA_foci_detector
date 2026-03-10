
import napari
import numpy as np
from magicgui import magicgui
from pathlib import Path

from src.detector import run_detection
from src.utils import save_counts_csv
from typing import Sequence

from napari.qt.threading import thread_worker


# store last detection result
state = {}


@thread_worker(progress=True)
def run_batch_detection(file_paths):

    total = len(file_paths)
    results = []

    for i, file_path in enumerate(file_paths):

        result = run_detection(str(file_path))
        results.append(result)
        yield int((i + 1) / total * 100)   # progress %

    return results



@magicgui(
    file_path={"label": "LSM Files", "mode": "rm"},
    call_button="Run Detection",
)
def detect_spots(file_path: Sequence[Path]):

    if not file_path:
        print("Please select LSM files.")
        return

    worker = run_batch_detection(file_path)

    def on_done(results):

        state["results"] = results
        last = results[-1]
        viewer.layers.clear()
        img = last.img   # (Z, C, Y, X)

        channel_names = ["DNA", "Foci", "Ch3", "Ch4"]
        colormaps = ["gray", "magenta", "green", "yellow"]

        for c in range(img.shape[1]):

            stack = img[:, c].astype(np.float32)

            viewer.add_image(
                stack,
                name=channel_names[c] if c < len(channel_names) else f"Channel {c}",
                colormap=colormaps[c] if c < len(colormaps) else "gray",
                rendering="mip",
                contrast_limits=(
                    np.percentile(stack, 1),
                    np.percentile(stack, 99.5)
                )
            )

        points = last.blobs_filtered[:, :3] if len(last.blobs_filtered) > 0 else np.empty((0,3))

        viewer.add_points(
            points,
            name="Detected Spots",
            size=8,
            face_color="red",
        )
        print("Batch detection finished.")

    worker.returned.connect(on_done)
    worker.start()



def handle_file_drop(viewer, paths):

    lsm_files = [p for p in paths if str(p).lower().endswith(".lsm")]

    if not lsm_files:
        print("No LSM files dropped.")
        return

    print(f"{len(lsm_files)} file(s) dropped.")
    worker = run_batch_detection(lsm_files)

    def on_done(results):

        state["results"] = results
        last = results[-1]
        viewer.layers.clear()
        stack = last.spot_stack

        viewer.add_image(
            stack,
            name="Foci channel",
            colormap="gray",
            rendering="mip",
            contrast_limits=(
                np.percentile(stack,1),
                np.percentile(stack,99.5)
            )
        )

        points = last.blobs_filtered[:, :3] if len(last.blobs_filtered) > 0 else np.empty((0,3))

        viewer.add_points(
            points,
            name="Detected Spots",
            size=8,
            face_color="red"
        )
        print("Detection finished.")

    worker.returned.connect(on_done)
    worker.start()




@magicgui(call_button="Export CSV")
def export_csv():

    if "results" not in state:
        print("Run detection first.")
        return

    csv_path = save_counts_csv(state["results"])
    print(f"CSV saved: {csv_path}")


# create napari viewer
viewer = napari.Viewer(ndisplay=3)

# enable drag-and-drop on the main window
qt_window = viewer.window._qt_window
qt_window.setAcceptDrops(True)

def dropEvent(event):
    paths = [url.toLocalFile() for url in event.mimeData().urls()]
    handle_file_drop(viewer, paths)
    event.accept()

qt_window.dropEvent = dropEvent


# add GUI panels
viewer.window.add_dock_widget(detect_spots, area="right")
viewer.window.add_dock_widget(export_csv, area="right")

napari.run()