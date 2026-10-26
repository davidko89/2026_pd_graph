## 2026-10-26 — Cell graph proof of concept (song2020_d274, PD02_B25-1-TH-2-2)

**Goal:** work through the full graph pipeline on one practice slide
(TH-DAB, day 274) to decide what to measure and how. Not a biological
result: one slide, no groups, distances in pixels.

### Pipeline as of today

slide → crop to graft oval (02c) → candidate detection, blob_log (02c)
→ manual ground truth (03a) → detector evaluation (03b, Steps 1–5)
→ Delaunay graph + pruning (03b, Steps 6–7) → metrics (03b, Step 8)
→ reusable module validated (03b, Step 9; `src/graph_metrics.py`)

### Scripts and files

| File | Env | Role |
|---|---|---|
| `scripts/02c_...` | pathology_gpu | Ring/oval crop + blob_log candidates. Added save of crop, mask, bbox. |
| `scripts/03a_annotate_ground_truth_song2020_d274.py` | **cv_gpu** | napari point annotation, run cell by cell; save is its own cell. |
| `scripts/03b_build_cell_graph_song2020_d274.py` | pathology_gpu | Steps 1–9: detector eval, graph build, pruning, metrics, module validation. Complete. |
| `src/graph_metrics.py` (+ `src/__init__.py`) | pathology_gpu | Reusable: `build_edges`, `tukey_cutoff`, `prune_edges`, `to_networkx`, `compute_metrics` (one summary row per slide). |

Outputs (`results/tables|figures/song2020_d274/`): ground_truth_points.csv,
detection_eval.csv, detection_vs_gt.png, graph_delaunay_vs_knn.png,
graph_pruned.png, graph_pruning_summary.csv, edges_*_pruned.csv,
node_metrics_delaunay.csv, graph_summary_delaunay.csv, radial_profile.csv,
graph_metrics_delaunay.png, graph_summary_module.csv.

### Environment notes

- napari is **not** installed in pathology_gpu: it would upgrade pydantic
  v1 → v2, and cellvit requires v1. Annotation runs in cv_gpu (napari 0.4.18).
- In a Jupyter/VS Code interactive window, `napari.run()` does not block, so a
  save placed after it runs immediately. The first annotation attempt was lost
  this way; 03a now saves from a separate cell.

### Ground truth

- Counting rule: visible TH+ soma with a defined cell-body outline; exclude
  fiber varicosities, dendrite segments, background specks.
- Seeded from the 79 larger-than-floor blobs: 12 removed, 53 added → **120 cells**.
- Caveat: seeding biases precision of the seeded set upward.

### Detector evaluation (blob_log vs ground truth, Hungarian matching)

- 68% of candidates (170/249) sat at the minimum blob radius (8.49 px =
  min_sigma·√2); visually fiber chains and specks. Only ~11 were real cells.
- radius > floor, tol 15 px: **precision 0.86, recall 0.57** (TP 68, FP 11, FN 52).
- All 249 candidates: recall at most 0.69 → ~40 cells never detected at all.
- Misses cluster in faint upper-left somata; FPs sit on the fiber ring
  (20-iteration dilation boundary).
- **Decision:** graph nodes = 120 annotated points, not detector output.

### Graph construction

- Compared Delaunay (343 edges) vs kNN k=6 (447 edges).
- Pruning cutoff = Tukey fence (Q3 + 1.5·IQR) of Delaunay edge lengths
  = **327 px** (Q1 77, Q3 177). Removed 28 / 30 long edges.
- After pruning both graphs: 2 components (117 + 3 cells), 0 isolated cells.
  Jaccard edge overlap 0.62 (same large-scale shape, different local edges).
- **Decision:** pruned Delaunay is the primary graph (parameter-free, planar,
  classical cell-graph definition).

### Metrics (pruned Delaunay)

| Metric | Value | Note |
|---|---|---|
| mean degree | 5.25 | ~6 expected for any Delaunay graph; low on network edge |
| mean clustering / transitivity | 0.468 / 0.404 | Random points in a synthetic oval gave 0.466 → mostly triangulation, not biology |
| components | 2 (117 + 3) | frac_largest_component 0.975 |
| mean NND | 70.5 px | CSR expectation 80.4 px |
| Clark–Evans R (z) | 0.877 (−2.58) | Mildly clustered within the oval; no edge correction |
| core_enrichment | 0.79 | Inner half of graft holds ~21% fewer cells than its area share |
| centroid offset | 0.25 of graft radius at 103° | Cells shifted upward |

Radial profile: nearly empty core (<167 px), peak at 335–669 px, decline
beyond. Outer rings of an elongated oval only cover its top/bottom ends,
so the radial profile cannot separate top from bottom bias.

Interpretation for this slide: connected horseshoe-shaped TH+ network around
a nearly cell-free core, weighted to the upper half. Observation only.

Drop for future: graph density (driven by cell count).

### Module validation (03b Step 9)

`src/graph_metrics.py` (scipy Delaunay, no squidpy) reproduced 03b exactly:
identical pruned edge set, all scalar metrics, identical radial counts.

---

## Open items and future plans

### Before comparing real slides
- [ ] Get scanner µm/px calibration (embedded DPI tags are placeholders).
- [ ] Fix ONE pruning cutoff in µm for all slides (pooled `tukey_cutoff()` or
      biological rationale). 327 px is provisional. Never re-estimate per slide.
- [ ] Re-check 02c dilation (20 iterations): FPs on the fiber ring.

### Detector (replaces manual annotation at scale)
- [ ] Choose a model (fine-tuned CellViT or an alternative); first check which
      label type each needs (points vs masks).
- [ ] Annotate several slides with the same counting rule; hold out test slides.
- [ ] Benchmark with the 03b matching: must beat precision 0.86 / recall 0.57.
- [ ] At scale: spot-check annotation of a subset per staining batch.
- [ ] Note: future z-stack / multiplex IF images will likely need a different
      detector than DAB brightfield.

### Multi-slide analysis (`scripts/04_...`)
- [ ] Loop `compute_metrics()` over slides; attach animal ID, section, group.
- [ ] Unit of analysis = animal: aggregate sections per animal (or mixed model,
      section nested in animal). Never treat cells or slides as independent.
- [ ] Primary comparison: good vs poor responders (graft graph exists only in
      grafted animals). Also correlate metrics with continuous behavioral
      recovery (more power with small subgroups).
- [ ] Pre-specify primary metrics (e.g., core_enrichment, Clark–Evans R,
      frac_largest_component, mean NND in µm); correct for multiple comparisons.
- [ ] Optional: Monte Carlo CSR within each graft mask for null-normalized
      metrics, if graft shapes differ a lot between animals.
- [ ] Keep acquisition consistent: section level, staining batch, scanner settings.

### Housekeeping
- [ ] README: add `src/` to folder description; add napari/cv_gpu note.
- [ ] Commit `src/`, 03a, 03b.
