#%%
from pathlib import Path
import tifffile

# --- Path setup -------------------------------------------------------------

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"
RAW_DATA_ROOT = DATA_ROOT / "raw_data" / "song2020"
PREPROC_DATA_DIR = DATA_ROOT / "preproc_data" / "song2020_d028"

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
import subprocess

# --- Convert one preprocessed file to pyramidal TIFF (required by OpenSlide/CellViT) ---

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

# Test on the single sample file we already extracted
test_input = PREPROC_DATA_DIR / f"{sample_file.stem.replace(' ', '_')}_page0.tif"
pyramid_out = convert_to_pyramid(test_input, PREPROC_DATA_DIR)
print("Pyramid TIFF written to:", pyramid_out)
print("File size:", pyramid_out.stat().st_size, "bytes")

#%%
import openslide

# --- Verify OpenSlide can open the pyramid TIFF -------------------------------

slide = openslide.OpenSlide(str(pyramid_out))

print("Dimensions (level 0):", slide.dimensions)
print("Level count:", slide.level_count)
print("Level dimensions:", slide.level_dimensions)
print("Level downsamples:", slide.level_downsamples)

# Check what metadata OpenSlide can find (this tells us whether it picked up
# any pixel-size/mpp info — expect this to be empty/absent given our known
# unreliable DPI tag, but worth confirming rather than assuming)
print("\nAvailable properties:")
for key in slide.properties.keys():
    if "mpp" in key.lower() or "resolution" in key.lower():
        print(f"  {key}: {slide.properties[key]}")

mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, None)
mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, None)
print(f"\nOpenSlide MPP-X: {mpp_x} | MPP-Y: {mpp_y}")

slide.close()

#%%
# --- Convert all 32 preprocessed files to pyramidal TIFFs --------------------

pyramid_results = []

for f in tif_files:
    page0_path = PREPROC_DATA_DIR / f"{f.stem.replace(' ', '_')}_page0.tif"
    pyramid_path = convert_to_pyramid(page0_path, PREPROC_DATA_DIR)
    pyramid_results.append({
        "source": page0_path.name,
        "pyramid_output": pyramid_path.name,
        "size_bytes": pyramid_path.stat().st_size,
    })

print(f"Converted {len(pyramid_results)} files to pyramidal TIFF.")
