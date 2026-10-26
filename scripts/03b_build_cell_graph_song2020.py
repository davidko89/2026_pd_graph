#%%
# Builds a spatial graph (no GNN) from candidate TH+ blob coordinates
# produced by 02c. Graph construction follows squidpy conventions
# (Palla et al. 2022); edge definitions compared: Delaunay vs kNN.
#
# RUN IN: pathology_gpu kernel.
# Inputs:  02c -> detected_cells.csv, crop_img.tif, oval_mask.npy
#          03a -> ground_truth_points.csv (manual annotation, cv_gpu/napari)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import anndata as ad
import squidpy as sq
import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path.home() / "projects" / "2026_pd_graph"
DATA_ROOT = Path.home() / "data" / "2026_pd_graph"

PROC_DATA_DIR = DATA_ROOT / "proc_data" / "song2020_d274"
RESULTS_FIG_DIR = PROJECT_ROOT / "results" / "figures" / "song2020_d274"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "song2020_d274"

SAMPLE_ID = "PD02_B25-1-TH-2-2"
blob_csv = RESULTS_TABLE_DIR / f"{SAMPLE_ID}_detected_cells.csv"
crop_tif = PROC_DATA_DIR / f"{SAMPLE_ID}_crop_img.tif"
mask_npy = PROC_DATA_DIR / f"{SAMPLE_ID}_oval_mask.npy"
gt_csv = RESULTS_TABLE_DIR / f"{SAMPLE_ID}_ground_truth_points.csv"

for p in [blob_csv, crop_tif, mask_npy, gt_csv]:
    print(p.name, "exists:", p.exists())
print("squidpy", sq.__version__, "| networkx", nx.__version__)

#%%
# --- Step 2: load candidates and look at them -------------------------------

df = pd.read_csv(blob_csv)
crop_img = tifffile.imread(crop_tif)
oval_mask = np.load(mask_npy)
print(f"n candidates: {len(df)} | crop: {crop_img.shape} | mask: {oval_mask.shape}")
print(df["radius_px"].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 10),
                         gridspec_kw={"width_ratios": [2, 1]})
axes[0].imshow(crop_img)
axes[0].scatter(df["x"], df["y"], s=8, c="lime")
axes[0].set_title(f"Candidate nodes (n={len(df)})")
axes[0].axis("off")

axes[1].hist(df["radius_px"], bins=30)
axes[1].set_xlabel("blob radius (px)")
axes[1].set_title("Node size distribution")
fig.tight_layout()
plt.show()

#%%
# --- Step 3: split candidates by blob scale ---------------------------------
# Radius takes discrete values (fixed by num_sigma), so count them directly.
# ~68% sit at the floor radius (min_sigma * sqrt(2)); visually these track
# fiber chains and specks, while larger blobs track somata.

print(df["radius_px"].round(2).value_counts().sort_index())

floor_r = df["radius_px"].min()
is_floor = np.isclose(df["radius_px"], floor_r)
print(f"floor radius = {floor_r:.2f} px | at floor: {is_floor.sum()} | larger: {(~is_floor).sum()}")

fig, ax = plt.subplots(figsize=(9, 18))
ax.imshow(crop_img)
ax.scatter(df.loc[is_floor, "x"], df.loc[is_floor, "y"],
           s=14, facecolors="none", edgecolors="red", linewidths=1,
           label=f"radius = floor ({is_floor.sum()})")
ax.scatter(df.loc[~is_floor, "x"], df.loc[~is_floor, "y"],
           s=40, facecolors="none", edgecolors="lime", linewidths=1.5,
           label=f"radius > floor ({(~is_floor).sum()})")
ax.legend(loc="upper right")
ax.set_title("Candidate nodes by blob scale")
ax.axis("off")
plt.savefig(RESULTS_FIG_DIR / f"{SAMPLE_ID}_nodes_by_scale.png",
            dpi=150, bbox_inches="tight")
plt.show()

#%%
# --- Step 4: precision / recall of candidates vs manual ground truth --------
# One-to-one Hungarian matching under a distance tolerance (standard for
# point-based cell detection evaluation), so several detections on one soma
# can't all count as hits. Several tolerances so the result doesn't hinge on
# one cutoff. Note: GT was seeded from the radius>floor set, which biases
# that set's precision upward -- recall is the more informative number.

gt = pd.read_csv(gt_csv)
gt_yx = gt[["y", "x"]].to_numpy()
print(f"Ground-truth cells: {len(gt)}")

def match_points(pred_yx, gt_yx, tol_px):
    """One-to-one Hungarian matching; pairs farther than tol_px don't count."""
    d = cdist(pred_yx, gt_yx)
    rows, cols = linear_sum_assignment(d)
    tp = int((d[rows, cols] <= tol_px).sum())
    precision = tp / len(pred_yx)
    recall = tp / len(gt_yx)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return tp, precision, recall, f1

candidate_sets = {
    "all_candidates": df[["y", "x"]].to_numpy(),
    "radius_gt_floor": df.loc[~is_floor, ["y", "x"]].to_numpy(),
}

rows_out = []
for name, pred in candidate_sets.items():
    for tol in [10, 15, 20]:
        tp, p, r, f1 = match_points(pred, gt_yx, tol)
        rows_out.append({"set": name, "n_pred": len(pred), "tol_px": tol,
                         "TP": tp, "precision": round(p, 3),
                         "recall": round(r, 3), "F1": round(f1, 3)})

eval_df = pd.DataFrame(rows_out)
print(eval_df.to_string(index=False))
eval_df.to_csv(RESULTS_TABLE_DIR / f"{SAMPLE_ID}_detection_eval.csv", index=False)

#%%
# --- Step 5: visualize TP / FP / FN for radius>floor set (tol = 15 px) -------
# Result 2026-10-26: TP 68 | FP 11 | FN 52. Misses cluster in the faint
# upper-left somata; FPs sit on the fiber ring (dilation boundary).
# DECISION: detector recall (0.57) is too low to build graphs on, so graph
# nodes for this slide = the 120 manually annotated ground-truth points.

TOL_PX = 15
pred_yx = df.loc[~is_floor, ["y", "x"]].to_numpy()

d = cdist(pred_yx, gt_yx)
rows, cols = linear_sum_assignment(d)
ok = d[rows, cols] <= TOL_PX
tp_pred, tp_gt = rows[ok], cols[ok]

fp_mask = np.ones(len(pred_yx), bool); fp_mask[tp_pred] = False
fn_mask = np.ones(len(gt_yx), bool);   fn_mask[tp_gt] = False
print(f"TP {len(tp_pred)} | FP {fp_mask.sum()} | FN (missed) {fn_mask.sum()}")

fig, ax = plt.subplots(figsize=(9, 18))
ax.imshow(crop_img)
ax.scatter(pred_yx[tp_pred, 1], pred_yx[tp_pred, 0], s=40, facecolors="none",
           edgecolors="lime", linewidths=1.5, label=f"TP ({len(tp_pred)})")
ax.scatter(pred_yx[fp_mask, 1], pred_yx[fp_mask, 0], s=40, facecolors="none",
           edgecolors="red", linewidths=1.5, label=f"FP ({fp_mask.sum()})")
ax.scatter(gt_yx[fn_mask, 1], gt_yx[fn_mask, 0], s=40, marker="x",
           c="yellow", linewidths=1.5, label=f"FN / missed ({fn_mask.sum()})")
ax.legend(loc="upper right")
ax.set_title(f"Detector vs ground truth (radius > floor, tol {TOL_PX} px)")
ax.axis("off")
plt.savefig(RESULTS_FIG_DIR / f"{SAMPLE_ID}_detection_vs_gt.png",
            dpi=150, bbox_inches="tight")
plt.show()

#%%
# ============================================================================
# GRAPH CONSTRUCTION (nodes = 120 ground-truth TH+ somata from 03a)
# ============================================================================
# --- Step 6: build Delaunay and kNN graphs, compare edge lengths -------------
# Two standard edge definitions in cell-graph histology:
#   Delaunay: natural neighbors, no parameter; long edges across empty space
#             (esp. on the outer hull) must be pruned later.
#   kNN (k=6): each cell links to its 6 nearest; k=5-6 is common practice.
# Squidpy stores both as sparse adjacency + distance matrices in an AnnData,
# the same object type used for scRNA-seq. Distances are in PIXELS until the
# scanner's true um/px is known.

from matplotlib.collections import LineCollection
from scipy import sparse

nodes = gt[["y", "x"]].to_numpy()
adata = ad.AnnData(X=np.zeros((len(nodes), 0), dtype=np.float32))
adata.obs_names = [f"cell_{i:03d}" for i in range(len(nodes))]
adata.obsm["spatial"] = nodes[:, ::-1].copy()   # squidpy expects (x, y)

sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True,
                        key_added="delaunay")
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6,
                        key_added="knn6")

def edge_table(adata, key):
    """Undirected edge list (i, j, length_px) from a squidpy distance matrix."""
    D = adata.obsp[f"{key}_distances"].tocsr()
    D = D.maximum(D.T)                    # symmetrize (kNN is directional)
    D = sparse.triu(D, k=1).tocoo()       # each edge once
    return pd.DataFrame({"i": D.row, "j": D.col, "length_px": D.data})

edges = {key: edge_table(adata, key) for key in ["delaunay", "knn6"]}

for key, e in edges.items():
    deg = np.bincount(np.r_[e["i"], e["j"]], minlength=adata.n_obs)
    q = e["length_px"].quantile([0.5, 0.9, 0.99])
    print(f"{key:9s} edges {len(e):4d} | mean degree {deg.mean():.2f} | "
          f"length median {q[0.5]:.0f}, p90 {q[0.9]:.0f}, p99 {q[0.99]:.0f}, "
          f"max {e['length_px'].max():.0f} px")

fig, axes = plt.subplots(1, 3, figsize=(20, 14),
                         gridspec_kw={"width_ratios": [1, 1, 1.2]})
xy = adata.obsm["spatial"]
for ax, key in zip(axes[:2], ["delaunay", "knn6"]):
    e = edges[key]
    segs = np.stack([xy[e["i"]], xy[e["j"]]], axis=1)
    ax.imshow(crop_img)
    ax.add_collection(LineCollection(segs, colors="cyan", linewidths=0.8))
    ax.scatter(xy[:, 0], xy[:, 1], s=10, c="yellow", zorder=3)
    ax.set_title(f"{key}: {len(e)} edges")
    ax.axis("off")

bins = np.linspace(0, max(e["length_px"].max() for e in edges.values()), 50)
for key, e in edges.items():
    axes[2].hist(e["length_px"], bins=bins, alpha=0.6, label=key)
axes[2].set_xlabel("edge length (px)")
axes[2].set_ylabel("count")
axes[2].set_title("Edge length distribution")
axes[2].legend()
fig.tight_layout()
plt.savefig(RESULTS_FIG_DIR / f"{SAMPLE_ID}_graph_delaunay_vs_knn.png",
            dpi=150, bbox_inches="tight")
plt.show()

#%%
# --- Step 7: prune long edges with a data-driven cutoff ----------------------
# Cutoff = Tukey outlier fence (Q3 + 1.5*IQR) of the DELAUNAY edge lengths.
# Delaunay is used as the reference because it has no free parameter (kNN's
# length distribution depends on the chosen k). The SAME cutoff is applied to
# both graphs so they can be compared fairly.
# After pruning we report: edges removed, isolated cells (degree 0), number of
# connected components, and how much the two pruned edge sets overlap
# (Jaccard). High overlap = the edge definition matters little for this data.

L = edges["delaunay"]["length_px"]
q1, q3 = L.quantile([0.25, 0.75])
CUTOFF_PX = q3 + 1.5 * (q3 - q1)
print(f"Delaunay Q1 {q1:.0f} | Q3 {q3:.0f} | cutoff (Q3 + 1.5 IQR) = {CUTOFF_PX:.0f} px")

pruned = {k: e[e["length_px"] <= CUTOFF_PX].reset_index(drop=True)
          for k, e in edges.items()}

summary = []
graphs = {}
for key in ["delaunay", "knn6"]:
    e = pruned[key]
    G = nx.Graph()
    G.add_nodes_from(range(adata.n_obs))
    G.add_weighted_edges_from(zip(e["i"], e["j"], e["length_px"]),
                              weight="length_px")
    graphs[key] = G
    deg = np.array([d for _, d in G.degree()])
    summary.append({
        "graph": key,
        "edges_before": len(edges[key]),
        "edges_after": len(e),
        "removed": len(edges[key]) - len(e),
        "isolated_cells": int((deg == 0).sum()),
        "components": nx.number_connected_components(G),
        "largest_component": len(max(nx.connected_components(G), key=len)),
        "mean_degree": round(deg.mean(), 2),
    })
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

edge_sets = {k: set(zip(e["i"], e["j"])) for k, e in pruned.items()}
shared = edge_sets["delaunay"] & edge_sets["knn6"]
jaccard = len(shared) / len(edge_sets["delaunay"] | edge_sets["knn6"])
print(f"Shared edges: {len(shared)} | Jaccard overlap: {jaccard:.2f}")

summary_df.assign(cutoff_px=round(CUTOFF_PX, 1), jaccard=round(jaccard, 3)) \
    .to_csv(RESULTS_TABLE_DIR / f"{SAMPLE_ID}_graph_pruning_summary.csv", index=False)
for key, e in pruned.items():
    e.to_csv(RESULTS_TABLE_DIR / f"{SAMPLE_ID}_edges_{key}_pruned.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(20, 14),
                         gridspec_kw={"width_ratios": [1, 1, 1.2]})
for ax, key in zip(axes[:2], ["delaunay", "knn6"]):
    e = pruned[key]
    segs = np.stack([xy[e["i"]], xy[e["j"]]], axis=1)
    iso = np.array([graphs[key].degree(n) == 0 for n in range(adata.n_obs)])
    ax.imshow(crop_img)
    ax.add_collection(LineCollection(segs, colors="cyan", linewidths=0.8))
    ax.scatter(xy[~iso, 0], xy[~iso, 1], s=10, c="yellow", zorder=3)
    ax.scatter(xy[iso, 0], xy[iso, 1], s=40, facecolors="none",
               edgecolors="red", linewidths=1.5, zorder=4,
               label=f"isolated ({iso.sum()})")
    ax.legend(loc="upper right")
    ax.set_title(f"{key} pruned (<= {CUTOFF_PX:.0f} px): {len(e)} edges")
    ax.axis("off")

for key, e in edges.items():
    axes[2].hist(e["length_px"], bins=bins, alpha=0.6, label=key)
axes[2].axvline(CUTOFF_PX, color="k", ls="--", label=f"cutoff {CUTOFF_PX:.0f} px")
axes[2].set_xlabel("edge length (px)")
axes[2].set_ylabel("count")
axes[2].set_title("Edge lengths with pruning cutoff")
axes[2].legend()
fig.tight_layout()
plt.savefig(RESULTS_FIG_DIR / f"{SAMPLE_ID}_graph_pruned.png",
            dpi=150, bbox_inches="tight")
plt.show()

#%%
# --- Step 8: graph + spatial metrics on the pruned Delaunay graph -----------
# Per cell: degree, clustering coefficient, mean edge length, nearest-
# neighbor distance (NND), distance from graft center.
# Whole graph: size, density, mean degree, mean clustering, components.
# Classical point-pattern stats: Clark-Evans R (mean NND / expected NND under
# complete spatial randomness; R < 1 clustered, ~1 random, > 1 regular) and a
# radial density profile (cells per area in rings from the graft center).
# Graft center = centroid of the graft-interior MASK, not of the cells: the
# cells form a horseshoe, so their own centroid is pulled toward the dense side.
# All distances in PIXELS until the scanner's um/px is known.
#
# Result 2026-10-26: connected horseshoe (117 of 120 cells) around a nearly
# empty core, upper-half heavy; Clark-Evans R = 0.88 (mildly clustered within
# the oval). Degree / clustering mostly reflect the triangulation itself.

from scipy.spatial import cKDTree
from scipy import ndimage

G = graphs["delaunay"]
xy = adata.obsm["spatial"]            # (x, y) in px
n = adata.n_obs

cy, cx = ndimage.center_of_mass(oval_mask)
area_px = int(oval_mask.sum())

nnd = cKDTree(xy).query(xy, k=2)[0][:, 1]   # k=1 is the cell itself
clust = nx.clustering(G)

comp_id = {}
for c_i, comp in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
    for v in comp:
        comp_id[v] = c_i

nodes_df = pd.DataFrame({
    "cell": adata.obs_names,
    "x": xy[:, 0],
    "y": xy[:, 1],
    "degree": [G.degree(v) for v in range(n)],
    "clustering": [clust[v] for v in range(n)],
    "mean_edge_len_px": [np.mean([d["length_px"] for _, _, d in G.edges(v, data=True)])
                         if G.degree(v) else np.nan for v in range(n)],
    "nnd_px": nnd,
    "dist_center_px": np.hypot(xy[:, 0] - cx, xy[:, 1] - cy),
    "component": [comp_id[v] for v in range(n)],
})

expected_nnd = 0.5 / np.sqrt(n / area_px)
clark_evans_R = nnd.mean() / expected_nnd

graph_summary = pd.DataFrame([{
    "n_cells": n,
    "n_edges": G.number_of_edges(),
    "graph_density": round(nx.density(G), 4),
    "mean_degree": round(nodes_df["degree"].mean(), 2),
    "mean_clustering": round(nx.average_clustering(G), 3),
    "transitivity": round(nx.transitivity(G), 3),
    "n_components": nx.number_connected_components(G),
    "largest_component": len(max(nx.connected_components(G), key=len)),
    "mean_nnd_px": round(nnd.mean(), 1),
    "expected_nnd_csr_px": round(expected_nnd, 1),
    "clark_evans_R": round(clark_evans_R, 3),
    "cells_per_1e5_px2": round(n / area_px * 1e5, 3),
    "cutoff_px": round(CUTOFF_PX, 1),
}])
print(graph_summary.T.to_string(header=False))

yy, xx = np.nonzero(oval_mask)
r_mask = np.hypot(xx - cx, yy - cy)
r_bins = np.linspace(0, r_mask.max(), 11)
ring_area, _ = np.histogram(r_mask, bins=r_bins)
ring_cells, _ = np.histogram(nodes_df["dist_center_px"], bins=r_bins)
ring_density = ring_cells / np.maximum(ring_area, 1) * 1e5
mean_density = n / area_px * 1e5

radial_df = pd.DataFrame({
    "r_inner_px": r_bins[:-1].round(0),
    "r_outer_px": r_bins[1:].round(0),
    "n_cells": ring_cells,
    "ring_area_px": ring_area,
    "cells_per_1e5_px2": ring_density.round(3),
})
print(radial_df.to_string(index=False))

nodes_df.to_csv(RESULTS_TABLE_DIR / f"{SAMPLE_ID}_node_metrics_delaunay.csv", index=False)
graph_summary.to_csv(RESULTS_TABLE_DIR / f"{SAMPLE_ID}_graph_summary_delaunay.csv", index=False)
radial_df.to_csv(RESULTS_TABLE_DIR / f"{SAMPLE_ID}_radial_profile.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(20, 12),
                         gridspec_kw={"width_ratios": [1, 1.1, 1.1]})

e = pruned["delaunay"]
segs = np.stack([xy[e["i"]], xy[e["j"]]], axis=1)
axes[0].imshow(crop_img)
axes[0].add_collection(LineCollection(segs, colors="white", linewidths=0.5, alpha=0.6))
sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=nodes_df["degree"], cmap="viridis",
                     s=30, zorder=3, edgecolors="k", linewidths=0.3)
axes[0].scatter(cx, cy, marker="+", s=200, c="red", zorder=4, label="graft center")
fig.colorbar(sc, ax=axes[0], fraction=0.04, label="degree")
axes[0].legend(loc="upper right")
axes[0].set_title("Pruned Delaunay graph, nodes colored by degree")
axes[0].axis("off")

r_mid = (r_bins[:-1] + r_bins[1:]) / 2
axes[1].plot(r_mid, ring_density, "o-")
axes[1].axhline(mean_density, color="k", ls="--", label="overall mean (uniform)")
axes[1].set_xlabel("distance from graft center (px)")
axes[1].set_ylabel("cells per 10^5 px^2")
axes[1].set_title("Radial density profile")
axes[1].legend()

axes[2].hist(nnd, bins=25)
axes[2].axvline(nnd.mean(), color="k", ls="-", label=f"observed mean {nnd.mean():.0f} px")
axes[2].axvline(expected_nnd, color="k", ls="--", label=f"random (CSR) {expected_nnd:.0f} px")
axes[2].set_xlabel("nearest-neighbor distance (px)")
axes[2].set_ylabel("cells")
axes[2].set_title(f"NND distribution (Clark-Evans R = {clark_evans_R:.2f})")
axes[2].legend()

fig.tight_layout()
plt.savefig(RESULTS_FIG_DIR / f"{SAMPLE_ID}_graph_metrics_delaunay.png",
            dpi=150, bbox_inches="tight")
plt.show()

#%%
# ============================================================================
# VALIDATION OF THE REUSABLE MODULE
# ============================================================================
# --- Step 9: check src/graph_metrics.py reproduces Steps 7-8 ----------------
# The module rebuilds the graph with scipy (no squidpy/AnnData) and must give
# the SAME edges and SAME metrics as above. If all checks pass, this script is
# complete and multi-slide work uses the module from a new script (04_...).
# CUTOFF_PX (327 px) is PROVISIONAL: for real cross-slide work, fix one cutoff
# in um (pooled data or biological rationale) once scanner calibration is known.

import sys
import importlib
sys.path.insert(0, str(PROJECT_ROOT))
import src.graph_metrics as gm
importlib.reload(gm)   # picks up edits to the module without a kernel restart

summary, nodes_mod, radial_mod, edges_mod = gm.compute_metrics(
    gt[["y", "x"]].to_numpy(), oval_mask, cutoff_px=CUTOFF_PX,
    method="delaunay", sample_id=SAMPLE_ID)

# 1) identical pruned edge set (squidpy, Step 7) vs module (scipy)
ref_edges = {(int(i), int(j)) for i, j in zip(pruned["delaunay"]["i"], pruned["delaunay"]["j"])}
mod_edges = {(int(i), int(j)) for i, j in zip(edges_mod["i"], edges_mod["j"])}
print("edge sets identical:", ref_edges == mod_edges)

# 2) identical scalar metrics (Step 8 values were rounded, hence the tolerance)
ref = graph_summary.iloc[0]
for key in ["n_edges", "mean_degree", "mean_clustering", "transitivity",
            "n_components", "largest_component", "mean_nnd_px",
            "expected_nnd_csr_px", "clark_evans_R"]:
    ok = np.isclose(summary[key], ref[key], atol=0.051)
    print(f"{key:22s} step8 {ref[key]:>9} | module {summary[key]:9.3f} | {'OK' if ok else 'MISMATCH'}")

# 3) identical radial profile counts
print("radial counts identical:",
      np.array_equal(radial_mod["n_cells"].to_numpy(), radial_df["n_cells"].to_numpy()))

# New, comparison-ready metrics the module adds
for key in ["frac_largest_component", "clark_evans_z", "core_enrichment",
            "centroid_offset_norm", "centroid_offset_angle_deg"]:
    print(f"{key:26s} {summary[key]:.3f}")

pd.DataFrame([summary]).to_csv(
    RESULTS_TABLE_DIR / f"{SAMPLE_ID}_graph_summary_module.csv", index=False)
# %%
