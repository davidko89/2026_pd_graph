#%%
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"

# Input: pyramid TIFF produced by 01b_ (D274 TH sample, not song2020 hNuc)
PREPROC_DATA_DIR = DATA_ROOT / "preproc_data" / "song2020_d274"

# Output: segmentation results — new subfolder, separate from the D028 hNuc attempt
PROC_DATA_DIR = DATA_ROOT / "proc_data" / "song2020_d274"

RESULTS_FIG_DIR = PROJECT_ROOT / "results" / "figures" / "song2020_d274"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "song2020_d274"

VALIDATION_LOG = PROJECT_ROOT / "docs" / "validation_log.md"

# --- Sanity checks ------------------------------------------------------------

print("PREPROC_DATA_DIR:", PREPROC_DATA_DIR, "-> exists:", PREPROC_DATA_DIR.exists())

# Just the one file we've preprocessed so far — not globbing all pyramid files,
# since we haven't converted the other 6 TH images yet
test_slide = PREPROC_DATA_DIR / "2-2_page0_pyramid.tif"
print("test_slide exists:", test_slide.exists())

PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
print("Output dirs ready:", PROC_DATA_DIR, "|", RESULTS_FIG_DIR, "|", RESULTS_TABLE_DIR)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import tifffile

img = tifffile.imread(test_slide, key=0)
print("Image shape:", img.shape, "dtype:", img.dtype)

r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
brightness = (r + g + b) / 3
stain_mask = (brightness < 200) & ((r - b) > 40)

cleaned_mask = ndimage.binary_opening(stain_mask, structure=np.ones((3, 3)))
labeled_mask, n_features = ndimage.label(cleaned_mask)
print(f"Found {n_features} connected components")

sizes = ndimage.sum(cleaned_mask, labeled_mask, range(1, n_features + 1))
largest_label = np.argmax(sizes) + 1
largest_blob = (labeled_mask == largest_label)
print(f"Largest component size: {int(sizes.max())} pixels "
      f"({100*sizes.max()/cleaned_mask.size:.2f}% of image)")

ys, xs = np.where(largest_blob)

# --- Tune these to adjust how tight/loose the crop is -------------------------
pad_top, pad_bottom, pad_sides = 200, 200, 100  # was effectively 450/50/50 (via uniform 50 + tall bbox)

y_min = max(ys.min() - pad_top, 0)
y_max = min(ys.max() + pad_bottom, img.shape[0])
x_min = max(xs.min() - pad_sides, 0)
x_max = min(xs.max() + pad_sides, img.shape[1])
crop = img[y_min:y_max, x_min:x_max]
print(f"Crop shape: {crop.shape}, bbox: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")

fig, axes = plt.subplots(1, 2, figsize=(10, 12))
axes[0].imshow(img)
axes[0].set_title("D274 TH — full field")
axes[0].axis("off")
axes[1].imshow(crop, interpolation="nearest")
axes[1].set_title(f"Largest component crop\nshape: {crop.shape}")
axes[1].axis("off")
fig.tight_layout()
plt.savefig(RESULTS_FIG_DIR / "PD02_B25-1-TH-2-2_cropped_v1.png", dpi=150, bbox_inches="tight")
plt.show()

#%%
# --- Diagnostic: what did the largest component actually select? -------------

print(f"Full image shape: {img.shape}")
print(f"Detected bbox: x[{xs.min()}:{xs.max()}] (width {xs.max()-xs.min()} of {img.shape[1]})")
print(f"              y[{ys.min()}:{ys.max()}] (height {ys.max()-ys.min()} of {img.shape[0]})")

fig, axes = plt.subplots(1, 2, figsize=(10, 12))
axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(largest_blob, cmap="gray")
axes[1].set_title(f"Largest component mask\n({int(sizes.max())} px)")
axes[1].axis("off")
fig.tight_layout()
plt.savefig(RESULTS_FIG_DIR / "PD02_B25-1-TH-2-2_largest_component_check.png", dpi=150, bbox_inches="tight")
plt.show()

#%%
# --- Take the largest connected component OF the interior mask itself --------

interior_mask = cleaned_mask
interior_labeled, n_interior = ndimage.label(interior_mask)
print(f"Interior mask has {n_interior} separate components (the oval + scattered tiny holes)")

interior_sizes = ndimage.sum(interior_mask, interior_labeled, range(1, n_interior + 1))
largest_interior_label = np.argmax(interior_sizes) + 1
oval_mask = (interior_labeled == largest_interior_label)

print(f"True oval size: {int(interior_sizes.max())} pixels "
      f"({100*interior_sizes.max()/oval_mask.size:.2f}% of image)")

ys_oval, xs_oval = np.where(oval_mask)
pad = 30
y_min = max(ys_oval.min() - pad, 0)
y_max = min(ys_oval.max() + pad, img.shape[0])
x_min = max(xs_oval.min() - pad, 0)
x_max = min(xs_oval.max() + pad, img.shape[1])

crop = img[y_min:y_max, x_min:x_max]
print(f"Crop shape: {crop.shape}, bbox: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")

fig, axes = plt.subplots(1, 3, figsize=(15, 8))
axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(oval_mask, cmap="gray")
axes[1].set_title("True oval only (largest interior component)")
axes[1].axis("off")
axes[2].imshow(crop, interpolation="nearest")
axes[2].set_title(f"Cropped to oval\nshape: {crop.shape}")
axes[2].axis("off")
fig.tight_layout()
plt.savefig(RESULTS_FIG_DIR / "PD02_B25-1-TH-2-2_cropped_v2.png", dpi=150, bbox_inches="tight")
plt.show()

#%%
import subprocess
import tifffile as tf

# --- Save the crop as its own file, then pyramid-convert it -------------------

def convert_to_pyramid(src_path: Path, dest_dir: Path) -> Path:
    """Convert a plain TIFF to a pyramidal, tiled TIFF using libvips.
    Required because OpenSlide (used internally by cellvit-inference)
    cannot open plain, non-pyramidal TIFFs."""
    dest_path = dest_dir / f"{src_path.stem}_pyramid.tif"
    cmd = [
        "vips", "tiffsave", str(src_path), str(dest_path),
        "--tile", "--tile-width", "256", "--tile-height", "256",
        "--pyramid", "--compression", "jpeg", "--Q", "90",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("vips STDERR:", result.stderr)
        raise RuntimeError(f"vips conversion failed for {src_path.name}")
    return dest_path

crop_path = PREPROC_DATA_DIR / "PD02_B25-1-TH-2-2_cropped.tif"
tf.imwrite(crop_path, crop)
print("Saved crop to:", crop_path)

crop_pyramid = convert_to_pyramid(crop_path, PREPROC_DATA_DIR)
print("Pyramid version:", crop_pyramid)

#%%
import os

# --- Run cellvit-inference on the CROPPED, PYRAMID-CONVERTED TIFF -------------

log_path = Path("/tmp/cellvit_d274_crop_run3.log")  # matches the successful run

cmd = [
    "cellvit-inference",
    "--model", "HIPT",
    "--outdir", str(PROC_DATA_DIR),
    "--geojson",
    "--graph",
    "process_wsi",
    "--wsi_path", str(crop_pyramid),
    "--wsi_mpp", "0.5",
    "--wsi_magnification", "20",
]

# Fixes the OOM crash we hit — see validation_log for details
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Running command:")
print(" ".join(cmd))
print()

with open(log_path, "w") as log_file:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    log_file.write(result.stdout)

print("Return code:", result.returncode)
print(f"Full log written to: {log_path}")
print("\n--- Output (last 3000 chars) ---")
print(result.stdout[-3000:])

#%%
import json

geojson_path = PROC_DATA_DIR / "PD02_B25-1-TH-2-2_cropped_pyramid" / "cell_detection.geojson"
print("Exists:", geojson_path.exists())

with open(geojson_path) as f:
    detections = json.load(f)

print("Type of detections:", type(detections))
print("Number of items:", len(detections))
print("\nType of first item:", type(detections[0]))
print("First item content:")
print(json.dumps(detections[0], indent=2)[:1500])  # first 1500 chars, avoid flooding output

#%%
# --- Overlay all detected cell centroids on the actual image -----------------

all_points = []
for feature in detections:
    class_name = feature["properties"]["classification"]["name"]
    coords = feature["geometry"]["coordinates"]
    for x, y in coords:
        all_points.append((x, y, class_name))

print(f"Total points across all classes: {len(all_points)}")

xs_det = [p[0] for p in all_points]
ys_det = [p[1] for p in all_points]

# Read the crop image at full resolution (same pixel space as detections)
crop_img = tifffile.imread(crop_pyramid, key=0)

fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(crop_img)
ax.scatter(xs_det, ys_det, s=40, facecolors="none", edgecolors="lime", linewidths=1.5)
ax.set_title(f"Detected cells (n={len(all_points)}) overlaid on D274 TH crop\n"
             "(class labels ignored — PanNuke categories are not biologically meaningful here)")
ax.axis("off")
fig.tight_layout()

overlay_path = RESULTS_FIG_DIR / "PD02_B25-1-TH-2-2_cropped_cellvit_overlay.png"
fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
plt.show()
print("Saved to:", overlay_path)
# %%
