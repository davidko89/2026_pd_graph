
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

**Scope:** Extract page 0 (real image) from all 32 hNuc TIFFs for sample
PD04 E25 / Post-TP-D028 (4wk); drop embedded 1/12-scale thumbnail page.
No pixel values altered — pure extraction/repackaging.

**Findings:**
- 32/32 files processed, all passed exact-match verification
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
