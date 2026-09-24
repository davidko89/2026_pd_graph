#%%
from pathlib import Path
import tifffile
import subprocess

# --- Path setup -------------------------------------------------------------

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"
RAW_DATA_ROOT = DATA_ROOT / "raw_data" / "song2020"
PREPROC_DATA_DIR = DATA_ROOT / "preproc_data" / "song2020_d274"
VALIDATION_LOG = PROJECT_ROOT / "docs" / "validation_log.md"

TH_DIR = (
    RAW_DATA_ROOT / "Post-TP-D274" / "122223 PD02,3,4 TH"
    / "121923 PD02 B25-1,2,3,4 TH" / "PD02 B25-1 TH images"
)

# --- Sanity checks ------------------------------------------------------------

print("TH_DIR:", TH_DIR, "-> exists:", TH_DIR.exists())

tif_files = sorted(TH_DIR.glob("*.tif"))
print(f"Found {len(tif_files)} .tif files")
for f in tif_files:
    print("  -", f.name)

PREPROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
print("PREPROC_DATA_DIR ready:", PREPROC_DATA_DIR)

# Confirm our specific target of interest is in the list
target_file = TH_DIR / "2-2.tif"
print("\nTarget file (2-2.tif) exists:", target_file.exists())

#%%
# --- Preprocess one file: extract page 0, save as clean single-page TIFF ----

def extract_page0(src_path: Path, dest_dir: Path) -> Path:
    """Read page 0 (the real image, not the embedded thumbnail) and save
    as a clean single-page TIFF. Returns the output path."""
    arr = tifffile.imread(src_path, key=0)
    dest_path = dest_dir / f"{src_path.stem.replace(' ', '_')}_page0.tif"
    tifffile.imwrite(dest_path, arr)
    return dest_path

# Test on 2-2.tif specifically
out_path = extract_page0(target_file, PREPROC_DATA_DIR)
print("Wrote:", out_path)

original_page0 = tifffile.imread(target_file, key=0)
check = tifffile.imread(out_path)
print("Re-read shape:", check.shape, "dtype:", check.dtype)
print("Matches original page 0 array:", (check == original_page0).all())

#%%
import subprocess

# --- Convert 2-2.tif's page0 to pyramidal TIFF --------------------------------

def convert_to_pyramid(src_path: Path, dest_dir: Path) -> Path:
    """Convert a plain TIFF to a pyramidal, tiled TIFF using libvips."""
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

test_input = PREPROC_DATA_DIR / f"{target_file.stem}_page0.tif"
pyramid_out = convert_to_pyramid(test_input, PREPROC_DATA_DIR)
print("Pyramid TIFF written to:", pyramid_out)
print("File size:", pyramid_out.stat().st_size, "bytes")

#%%
import openslide

slide = openslide.OpenSlide(str(pyramid_out))
print("Dimensions (level 0):", slide.dimensions)
print("Level count:", slide.level_count)
print("Level dimensions:", slide.level_dimensions)

mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, None)
print("OpenSlide MPP-X (expect fake value, ignore):", mpp_x)
slide.close()

# %%
