#%%
# Manual ground-truth annotation of TH+ somata for one test crop.
#
# RUN IN: cv_gpu kernel (napari 0.4.18) -- NOT pathology_gpu.
# Installing napari into pathology_gpu would upgrade pydantic v1 -> v2,
# which cellvit depends on (checked 2026-10-23). Annotation only reads a TIFF
# and writes a CSV, so it runs in its own env; 03a reads the CSV it produces.
#
# HOW TO USE (interactive / Jupyter, run cell by cell):
#   Cell 1: paths + load      Cell 2: open napari window
#   -> annotate in the window (keep it OPEN)
#   Cell 3: save points       Cell 4: verify the save
# In the window: 2 = add point, 3 = select, Delete = remove, 4 = pan/zoom.
#
# Why saving is its own cell: in Jupyter the window does not block the
# kernel, so a save placed after opening the viewer would run immediately,
# before any annotation (edits were lost this way on 2026-10-26).
#
# Counting rule (also in docs/validation_log.md):
#   visible TH+ soma with a defined cell-body outline; exclude fiber
#   varicosities, dendrite segments and background specks.

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import tifffile
import napari

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"

PROC_DATA_DIR = DATA_ROOT / "proc_data" / "song2020_d274"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "song2020_d274"

SAMPLE_ID = "PD02_B25-1-TH-2-2"
crop_tif = PROC_DATA_DIR / f"{SAMPLE_ID}_crop_img.tif"
blob_csv = RESULTS_TABLE_DIR / f"{SAMPLE_ID}_detected_cells.csv"
gt_csv = RESULTS_TABLE_DIR / f"{SAMPLE_ID}_ground_truth_points.csv"

crop_img = tifffile.imread(crop_tif)

# Seed with the larger-than-floor blob candidates. (The existing GT file is
# just these seeds -- the 2026-10-26 edits were never saved -- so start fresh.)
df = pd.read_csv(blob_csv)
is_floor = np.isclose(df["radius_px"], df["radius_px"].min())
seed = df.loc[~is_floor, ["y", "x"]].to_numpy()  # napari uses (row, col)
print(f"Crop: {crop_img.shape} | seeded {len(seed)} points")

#%%
# --- Open the annotation window (leave it open while you work) -------------

viewer = napari.Viewer(title=f"Ground truth: {SAMPLE_ID}")
viewer.add_image(crop_img, name="crop", rgb=True)
pts = viewer.add_points(
    seed,
    name="gt_cells",
    size=20,
    face_color="transparent",
    edge_color="cyan",   # napari 0.4.x name (0.5+ renamed to border_color)
    edge_width=0.15,
)
pts.mode = "add"
print("Annotate now. When finished, press 4 and run the SAVE cell. "
      "Don't close the window first.")

#%%
# --- SAVE: run only after you finish annotating -----------------------------
# Re-run anytime to save progress; a .bak copy of the previous file is kept.

if gt_csv.exists():
    shutil.copy(gt_csv, gt_csv.with_suffix(".bak.csv"))
gt = pd.DataFrame(pts.data, columns=["y", "x"])
gt.to_csv(gt_csv, index=False)
print(f"Saved {len(gt)} ground-truth cells to: {gt_csv}")

#%%
# --- Verify the save actually contains your edits ---------------------------

seed_set = {tuple(p) for p in seed}
gt_set = {tuple(p) for p in pd.read_csv(gt_csv)[["y", "x"]].to_numpy()}
print("identical to seeds:", seed_set == gt_set, "(should be False)")
print("seeds removed:", len(seed_set - gt_set))
print("points added:", len(gt_set - seed_set))

# %%
