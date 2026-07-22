# DNA Foci Detector

A desktop tool for counting DNA damage foci from fluorescence microscopy `.lsm` z-stacks.

Counting DNA damage foci manually is slow and repetitive. Researchers often need to open 3D `.lsm` files in microscopy software, move through z-slices, identify nuclei, and count bright foci by hand. This tool automates that workflow and gives a visual overlay so user can inspect the result. The goal is to make routine foci counting faster and more consistent.


<img width="1352" height="741" alt="spots_detected_window" src="https://github.com/user-attachments/assets/309524ae-b8da-40e2-b78e-6d39a4849c2a" />


## Working

The pipeline has 2 stages - nucleus segmentation and spot detection

### Nucleus segmentation

The segmenter creates a 2D projection from the nucleus channel. It normalizes and smooths the image, then uses a bandpass-style response to enhance the nucleus boundary.
It searches over a range of circular/annular templates and selects the top two nucleus candidates. These are converted into 2D labels.
The 2D nucleus labels are then expanded into 3D masks using the z-intensity profile of each nucleus. The mask is softly tapered across z, with padding and radius scaling so that foci near the top, bottom, or edge of the nucleus are not removed too aggressively.

### Spot detection

The spot detector removes broad background haze, normalizes each z-slice, and reduces grain noise with median filtering. It then uses a multiscale Difference-of-Gaussians response to enhance compact bright foci. Candidate spots are extracted as 3D local maxima, filtered by size, local contrast, SNR, merged if duplicated, and assigned to the correct nucleus using the 3D label mask.
The final output is the foci count for each nucleus.


The app uses Napari, a Python-based viewer for multidimensional microscopy data. It lets the user inspect z-stacks, image channels, segmentation masks, and detected foci overlays in the same window.



