"""
00_inspect_raw_song2020.py

Step 1: Read-only structural inspection of raw Song et al. 2020 practice data.
No processing yet — just confirming paths resolve and files are where expected.

Starting scope: hNuc images only (nucleus segmentation target).
TH images (fiber network) are a separate, later task — not handled here.
"""
#%%
from pathlib import Path

# --- Path setup -------------------------------------------------------------

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"
RAW_DATA_ROOT = DATA_ROOT / "raw_data" / "song2020"

# Starting with one sample: PD04 E25, Post-TP-D028 (4 weeks), hNuc channel
HNUC_DIR = RAW_DATA_ROOT / "Post-TP-D028" / "PD04 E25" / "Images (hNuc, PD04 E25, 4 weeks)"

OUTPUT_FIG_DIR = PROJECT_ROOT / "results" / "figures" / "song2020"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "song2020"
VALIDATION_LOG = PROJECT_ROOT / "docs" / "validation_log.md"

# Defined now for visibility, but NOT written to in this script.
# 00_inspect_raw_* is read-only inspection only — the raw -> preproc
# transformation (and its output) belongs to 01_build_preproc_song2020.py.
PREPROC_DATA_DIR = DATA_ROOT / "preproc_data" / "song2020"

# --- Sanity checks ------------------------------------------------------------

print("PROJECT_ROOT:", PROJECT_ROOT, "-> exists:", PROJECT_ROOT.exists())
print("RAW_DATA_ROOT:", RAW_DATA_ROOT, "-> exists:", RAW_DATA_ROOT.exists())
print("HNUC_DIR:", HNUC_DIR, "-> exists:", HNUC_DIR.exists())
print("PREPROC_DATA_DIR (not yet created, reserved for 01_):", PREPROC_DATA_DIR)

if HNUC_DIR.exists():
    tif_files = sorted(HNUC_DIR.glob("*.tif"))
    print(f"Found {len(tif_files)} .tif files in HNUC_DIR")
    for f in tif_files[:3]:
        print("  -", f.name)
    if len(tif_files) > 3:
        print(f"  ... and {len(tif_files) - 3} more")
else:
    print("HNUC_DIR not found — check path/spacing above before proceeding.")

# Create output dirs now (safe even if we don't write to them yet)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
print("Output dirs ready:", OUTPUT_FIG_DIR, "|", OUTPUT_TABLE_DIR)

#%%
import tifffile
import numpy as np

# --- Open the first hNuc file and inspect it ---------------------------------

sample_file = tif_files[0]
print("Inspecting:", sample_file.name)

with tifffile.TiffFile(sample_file) as tif:
    n_pages = len(tif.pages)
    print("Number of pages/frames:", n_pages)

    for i, page in enumerate(tif.pages):
        print(f"\n--- Page {i} ---")
        print("  shape:", page.shape)
        print("  dtype:", page.dtype)
        # Resolution tags, if present (tells us µm/pixel — useful later)
        tags = page.tags
        if "XResolution" in tags and "ResolutionUnit" in tags:
            xres = tags["XResolution"].value
            yres = tags["YResolution"].value
            unit = tags["ResolutionUnit"].value
            print("  XResolution tag:", xres, "| YResolution tag:", yres, "| unit:", unit)
        else:
            print("  No resolution tags found on this page.")

    # Load the actual pixel data for page 0 for a quick numeric check
    arr = tif.pages[0].asarray()
    print("\nLoaded array shape:", arr.shape, "dtype:", arr.dtype)
    print("Pixel value range: min =", arr.min(), ", max =", arr.max())

#%%
import matplotlib.pyplot as plt

# --- Save and display a preview of page 0 ------------------------------------

preview_path = OUTPUT_FIG_DIR / f"{sample_file.stem}_preview.png"

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(arr)
ax.set_title(sample_file.name)
ax.axis("off")
fig.tight_layout()
fig.savefig(preview_path, dpi=150)
plt.show()

print("Preview saved to:", preview_path)

#%%
from datetime import date

log_entry = f"""
## {date.today().isoformat()} — 00_inspect_raw_song2020.py

**Scope:** Song et al. 2020 practice data, hNuc channel only, sample PD04 E25 / Post-TP-D028 (4wk).

**Findings:**
- {len(tif_files)} `.tif` files found in `HNUC_DIR`, one per XY field (XY01–XY32 naming).
- Each file has 2 TIFF pages: page 0 is the full image (1440x1920x3, uint8), page 1 is a
  1/12-scale embedded thumbnail (120x160x3) — not a second channel/z-slice. Only page 0
  is used going forward.
- Pixel value range on page 0: 31–232 (uint8, RGB, brightfield DAB IHC — not fluorescence
  despite `_CH4` in filename).
- **Caveat — pixel size unknown:** XResolution/YResolution tags read 96000/1000 = 96 DPI,
  which is a generic default-screen-resolution value baked in by export software, not a
  real microscope/objective calibration. Do NOT use this tag for physical distance
  conversion (µm/pixel). Treat all measurements as pixel-based only until real calibration
  data (scale bar or acquisition metadata) is available.
- Preview image saved and visually confirmed to match expected hNuc staining pattern
  (graft cell cluster along needle track).

**Status:** Raw data structure confirmed. Ready for `01_build_preproc_song2020.py`.
"""

VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
with open(VALIDATION_LOG, "a") as f:
    f.write(log_entry)

print("Logged to:", VALIDATION_LOG)
# %%
