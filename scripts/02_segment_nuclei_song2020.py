#%%
from pathlib import Path

# --- Path setup -------------------------------------------------------------

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"

# Input: pyramid TIFFs produced by 01_
PREPROC_DATA_DIR = DATA_ROOT / "preproc_data" / "song2020_d028"

# Output: segmentation results that 03_ (graph construction) will read from
PROC_DATA_DIR = DATA_ROOT / "proc_data" / "song2020_d028"

# Results: QC figures/tables, nothing downstream reads from these
RESULTS_FIG_DIR = PROJECT_ROOT / "results" / "figures" / "song2020_d028"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "song2020_d028"

VALIDATION_LOG = PROJECT_ROOT / "docs" / "validation_log.md"

# --- Sanity checks ------------------------------------------------------------

print("PREPROC_DATA_DIR:", PREPROC_DATA_DIR, "-> exists:", PREPROC_DATA_DIR.exists())

pyramid_files = sorted(PREPROC_DATA_DIR.glob("*_pyramid.tif"))
print(f"Found {len(pyramid_files)} pyramid TIFF files")
for f in pyramid_files[:3]:
    print("  -", f.name)
if len(pyramid_files) > 3:
    print(f"  ... and {len(pyramid_files) - 3} more")

# This script (02_) owns creating PROC_DATA_DIR, since it's the first
# script that writes segmentation output there.
PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
print("Output dirs ready:", PROC_DATA_DIR, "|", RESULTS_FIG_DIR, "|", RESULTS_TABLE_DIR)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import tifffile

# --- Read page 0 explicitly (avoids pyramid-page ambiguity warning) ----------
img = tifffile.imread(test_slide, key=0)

r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
brightness = (r + g + b) / 3
stain_mask = (brightness < 200) & ((r - b) > 40)

# --- Find the LARGEST connected blob, not just any matching pixel -----------
# Clean up small speckle noise first (scratches, dust) with a binary opening
cleaned_mask = ndimage.binary_opening(stain_mask, structure=np.ones((3, 3)))

labeled_mask, n_features = ndimage.label(cleaned_mask)
print(f"Found {n_features} connected components")

if n_features > 0:
    sizes = ndimage.sum(cleaned_mask, labeled_mask, range(1, n_features + 1))
    largest_label = np.argmax(sizes) + 1
    largest_blob = (labeled_mask == largest_label)
    print(f"Largest component size: {int(sizes.max())} pixels "
          f"({100*sizes.max()/cleaned_mask.size:.2f}% of image)")

    ys, xs = np.where(largest_blob)
    pad_top, pad_bottom, pad_sides = 450, 50, 75

    y_min_adj = max(ys.min() - pad_top, 0)
    y_max_adj = min(ys.max() + pad_bottom, img.shape[0])
    x_min_adj = max(xs.min() - pad_sides, 0)
    x_max_adj = min(xs.max() + pad_sides, img.shape[1])

    print(f"Tight bounding box: x[{x_min_adj}:{x_max_adj}], y[{y_min_adj}:{y_max_adj}]")
    crop = img[y_min_adj:y_max_adj, x_min_adj:x_max_adj]
else:
    print("No connected components found — check mask thresholds.")
    crop = img

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(img)
axes[0].set_title("Full field")
axes[0].axis("off")
axes[1].imshow(crop, interpolation="nearest")
axes[1].set_title(f"Tight crop, native pixels\nshape: {crop.shape}")
axes[1].axis("off")
fig.tight_layout()
crop_preview_path = RESULTS_FIG_DIR / "crop_preview.png"
plt.savefig(crop_preview_path, dpi=150, bbox_inches="tight")
plt.show()

#%%
import subprocess

# --- Run cellvit-inference on a single pyramid TIFF ---------------------------

test_slide = pyramid_files[0]
print("Testing segmentation on:", test_slide.name)

cmd = [
    "cellvit-inference",
    "--model", "HIPT",              # lighter model, better fit for 16GB VRAM
    "--outdir", str(PROC_DATA_DIR),
    "--geojson",                     # export boundaries for QuPath/visualization
    "--graph",                       # also generate cell-graph representation
    "process_wsi",
    "--wsi_path", str(test_slide),
    "--wsi_mpp", "0.25",             # explicit, deliberate assumption (see 01_ log)
]

print("Running command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=True, text=True)

print("Return code:", result.returncode)
print("\n--- STDOUT (last 3000 chars) ---")
print(result.stdout[-3000:])
print("\n--- STDERR (last 3000 chars) ---")
print(result.stderr[-3000:])

# %%
