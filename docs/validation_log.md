
## 2026-09-22 — 00_inspect_raw_song2020.py

**Scope:** Song et al. 2020 practice data, hNuc channel only, sample PD04 E25 / Post-TP-D028 (4wk).

**Findings:**
- 32 `.tif` files found in `HNUC_DIR`, one per XY field (XY01–XY32 naming).
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

## 2026-09-23 — 01_build_preproc_song2020.py

**Scope:** Build preprocessed hNuc data for sample PD04 E25 / Post-TP-D028
(4wk), 32 fields, in two sub-steps: (1) extract page 0 from each raw TIFF,
dropping the embedded 1/12-scale thumbnail page; (2) convert each page-0
TIFF to pyramidal, tiled format via libvips, required because OpenSlide
(used internally by cellvit-inference) cannot open plain non-pyramidal
TIFFs.

**Findings:**
- Page 0 extraction: 32/32 files passed exact-match verification against
  source page 0 (byte-for-byte identical arrays). No pixel values altered —
  pure extraction/repackaging. No visualization performed here — exact-match
  is strictly stronger evidence than a visual check for a lossless
  extraction.
- Pyramid conversion: 32/32 files converted successfully via
  `vips tiffsave ... --tile --pyramid`. Verified with OpenSlide on one file:
  dimensions (1920, 1440) match original exactly, 4 pyramid levels generated
  correctly.
- **Important — embedded MPP tag on pyramid files is fake, do not trust it,
  ever:** OpenSlide reports `openslide.mpp-x/y = 1000.0` microns/pixel on
  the converted files. This is vips' generic fallback (1 px/mm), not a real
  measurement, and is unrelated to (and different from) the original file's
  already-unreliable 96 DPI tag. This is NOT limited to cellvit-inference —
  any tool that opens these pyramid TIFFs (QuPath, other OpenSlide-based
  tools) will silently report wrong physical distances if it trusts this
  tag. Always override explicitly; never assume the file's own metadata
  is meaningful.

**Decision carried forward to 02_:** True pixel calibration (µm/pixel) is
unknown and unrecoverable from file metadata. `--wsi_mpp 0.25` will be
passed explicitly to cellvit-inference (matching the model's native
training resolution) — never left to auto-detection. This is a deliberate,
documented, uncalibrated sanity-check assumption for this practice round,
not a real measurement.

**Status:** Preprocessing complete for this sample (32/32 files, both
sub-steps). Ready for `02_segment_nuclei_song2020.py`.