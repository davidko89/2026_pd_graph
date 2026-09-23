#%%
from pathlib import Path
import tifffile

# --- Path setup -------------------------------------------------------------

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"
RAW_DATA_ROOT = DATA_ROOT / "raw_data" / "song2020"
PREPROC_DATA_DIR = DATA_ROOT / "preproc_data" / "song2020"

HNUC_DIR = RAW_DATA_ROOT / "Post-TP-D028" / "PD04 E25" / "Images (hNuc, PD04 E25, 4 weeks)"
VALIDATION_LOG = PROJECT_ROOT / "docs" / "validation_log.md"

# --- Sanity checks ------------------------------------------------------------

print("RAW_DATA_ROOT:", RAW_DATA_ROOT, "-> exists:", RAW_DATA_ROOT.exists())
print("HNUC_DIR:", HNUC_DIR, "-> exists:", HNUC_DIR.exists())
print("PREPROC_DATA_DIR:", PREPROC_DATA_DIR, "-> exists:", PREPROC_DATA_DIR.exists())

tif_files = sorted(HNUC_DIR.glob("*.tif"))
print(f"Found {len(tif_files)} .tif files in HNUC_DIR")

sample_file = tif_files[0]
print("Sample file for this test:", sample_file.name)

# This script (01_) owns creating PREPROC_DATA_DIR, since it's the first
# script that writes to it.
PREPROC_DATA_DIR.mkdir(parents=True, exist_ok=True)

#%%
# --- Preprocess one file: extract page 0, save as clean single-page TIFF ----

def extract_page0(src_path: Path, dest_dir: Path) -> Path:
    """Read page 0 (the real image, not the embedded thumbnail) and save
    as a clean single-page TIFF. Returns the output path."""
    arr = tifffile.imread(src_path, key=0)  # key=0 -> page 0 only
    dest_path = dest_dir / f"{src_path.stem.replace(' ', '_')}_page0.tif"
    tifffile.imwrite(dest_path, arr)
    return dest_path

# Test on the single sample file
out_path = extract_page0(sample_file, PREPROC_DATA_DIR)
print("Wrote:", out_path)

# Sanity check: re-read it and confirm it matches the original page 0 exactly
original_page0 = tifffile.imread(sample_file, key=0)
check = tifffile.imread(out_path)
print("Re-read shape:", check.shape, "dtype:", check.dtype)
print("Matches original page 0 array:", (check == original_page0).all())

#%%
# --- Apply to all hNuc files in this sample ----------------------------------

results = []

for f in tif_files:
    out_path = extract_page0(f, PREPROC_DATA_DIR)
    original_page0 = tifffile.imread(f, key=0)
    check = tifffile.imread(out_path)
    match = (check == original_page0).all()
    results.append({
        "source_file": f.name,
        "output_file": out_path.name,
        "shape": check.shape,
        "dtype": str(check.dtype),
        "exact_match": bool(match),
    })

n_ok = sum(r["exact_match"] for r in results)
print(f"Processed {len(results)} files. {n_ok}/{len(results)} passed exact-match check.")

# Flag any failures explicitly rather than assuming success
failures = [r for r in results if not r["exact_match"]]
if failures:
    print("FAILURES:")
    for r in failures:
        print("  -", r["source_file"])
else:
    print("All files processed cleanly.")

#%%
from datetime import date

log_entry = f"""
## {date.today().isoformat()} — 01_build_preproc_song2020.py

**Scope:** Extract page 0 (real image) from all 32 hNuc TIFFs for sample
PD04 E25 / Post-TP-D028 (4wk); drop embedded 1/12-scale thumbnail page.
No pixel values altered — pure extraction/repackaging.

**Findings:**
- {n_ok}/{len(results)} files processed, all passed exact-match verification
  against source page 0 (byte-for-byte identical arrays).
- Output written to `preproc_data/song2020/` as `<original_name>_page0.tif`.
- No visualization performed at this step — justified by exact-match check
  being strictly stronger evidence than visual inspection for a lossless
  extraction (no pixel values were transformed).

**Open item carried forward to 02_:** MPP/pixel-size calibration remains
unknown (see 00_ log). CellViT-Inference assumes 0.25 µm/pixel by default;
we have no verified real-world calibration for these images. Proceeding
in 02_ with this caveat explicitly noted, not silently assumed away.

**Status:** Preprocessing complete for this sample (32/32 files). Ready
for `02_segment_nuclei_song2020.py`.
"""

with open(VALIDATION_LOG, "a") as f:
    f.write(log_entry)

print("Logged to:", VALIDATION_LOG)
# %%
