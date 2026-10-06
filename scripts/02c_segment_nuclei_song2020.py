#%%
# CellViT-Inference (HIPT) was tried on this sample first, in 02b_, and
# abandoned there — its detections landed on fiber crossings, not the TH+
# somata (see docs/validation_log.md, 2026-10-06 entry, for that negative
# result and the overlay figure proving how far off it was).
#
# This script uses classical image processing instead, without any trained
# segmentation model: ring detection -> fill -> true oval interior -> blob_log.
# It is NOT yet a validated cell detector either — see the caveats in the
# final cell below before trusting the output coordinates.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
from scipy import ndimage
from skimage.feature import blob_log
from skimage.color import rgb2gray

# --- Path setup -------------------------------------------------------------

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"

PREPROC_DATA_DIR = DATA_ROOT / "preproc_data" / "song2020_d274"
PROC_DATA_DIR = DATA_ROOT / "proc_data" / "song2020_d274"
RESULTS_FIG_DIR = PROJECT_ROOT / "results" / "figures" / "song2020_d274"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "song2020_d274"
VALIDATION_LOG = PROJECT_ROOT / "docs" / "validation_log.md"

PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)

test_slide = PREPROC_DATA_DIR / "2-2_page0_pyramid.tif"
print("test_slide exists:", test_slide.exists())

#%%
# --- Detect the fiber ring, fill it, find the true oval interior -------------

full_img = tifffile.imread(test_slide, key=0)
print("Full image shape:", full_img.shape)

r, g, b = full_img[..., 0].astype(int), full_img[..., 1].astype(int), full_img[..., 2].astype(int)
brightness = (r + g + b) / 3
stain_mask = (brightness < 200) & ((r - b) > 40)
cleaned = ndimage.binary_opening(stain_mask, structure=np.ones((3, 3)))

ring_labeled, _ = ndimage.label(cleaned)
ring_sizes = ndimage.sum(cleaned, ring_labeled, range(1, ring_labeled.max() + 1))
ring_mask = (ring_labeled == (np.argmax(ring_sizes) + 1))

filled = ndimage.binary_fill_holes(ring_mask)
interior = filled & (~ring_mask)

interior_labeled, _ = ndimage.label(interior)
interior_sizes = ndimage.sum(interior, interior_labeled, range(1, interior_labeled.max() + 1))
oval_mask = (interior_labeled == (np.argmax(interior_sizes) + 1))

# Dilate slightly to recover real cells sitting right at the fiber boundary.
# iterations=20 was the value that worked in testing; bump to 40 if cells
# near the ring edge (e.g. far left-middle, far right-upper-middle) are
# still being dropped — check those two regions against the saved figure
# below before trusting this value.
oval_mask_expanded = ndimage.binary_dilation(oval_mask, iterations=20)

ys, xs = np.where(oval_mask_expanded)
pad = 30
y_min = max(ys.min() - pad, 0)
y_max = min(ys.max() + pad, full_img.shape[0])
x_min = max(xs.min() - pad, 0)
x_max = min(xs.max() + pad, full_img.shape[1])

crop_img = full_img[y_min:y_max, x_min:x_max]
interior_mask_cropped = oval_mask_expanded[y_min:y_max, x_min:x_max]

print(f"Crop shape: {crop_img.shape}, bbox: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")
# Sanity check: catches stale-variable bugs if the crop and mask ever
# get computed from different-shaped arrays.
print("Shapes match:", crop_img.shape[:2] == interior_mask_cropped.shape[:2])

#%%
# --- Blob detection, restricted to the (dilated) interior region -------------

gray = rgb2gray(crop_img)
inverted = 1 - gray

blobs = blob_log(inverted, min_sigma=6, max_sigma=15, num_sigma=10, threshold=0.12)
print(f"Blobs before interior filtering: {len(blobs)}")

filtered_blobs = [
    (y, x, s) for y, x, s in blobs
    if 0 <= int(y) < interior_mask_cropped.shape[0]
    and 0 <= int(x) < interior_mask_cropped.shape[1]
    and interior_mask_cropped[int(y), int(x)]
]
print(f"Blobs after interior filtering: {len(filtered_blobs)}")

fig, axes = plt.subplots(1, 2, figsize=(18, 12))
axes[0].imshow(crop_img)
axes[0].set_title(f"Cropped to oval\nshape: {crop_img.shape}")
axes[0].axis("off")

axes[1].imshow(crop_img)
for y, x, sigma in filtered_blobs:
    radius = sigma * np.sqrt(2)
    circle = plt.Circle((x, y), radius, color="lime", linewidth=1.5, fill=False)
    axes[1].add_patch(circle)
axes[1].set_title(f"Candidate blobs, dilated-interior filtered (n={len(filtered_blobs)})")
axes[1].axis("off")

fig.tight_layout()
plt.savefig(RESULTS_FIG_DIR / "PD02_B25-1-TH-2-2_final_blob_detection.png", dpi=150, bbox_inches="tight")
plt.show()

#%%
# --- Save candidate blob coordinates as a results table -----------------------
# NOTE: these are "candidate blobs," not confirmed TH+ cells. Without a
# trained/fine-tuned segmentation model, this classical blob_log approach is
# not validated either — zoomed-in checks showed some detections landing on
# dendrite segments and background specks. Treat this as a starting point for
# threshold/parameter tuning, not as a finished cell-count pipeline.

df = pd.DataFrame(filtered_blobs, columns=["y", "x", "sigma"])
df["radius_px"] = df["sigma"] * np.sqrt(2)
table_path = RESULTS_TABLE_DIR / "PD02_B25-1-TH-2-2_detected_cells.csv"
df.to_csv(table_path, index=False)
print(f"Saved {len(df)} candidate blob coordinates to: {table_path}")
# %%
