"""Spatial cell-graph construction and metrics (no GNN).

Reusable version of the exploratory Steps 6-8 in
scripts/03b_build_cell_graph_song2020_d274.py, validated there in Step 9
against the original squidpy-based code on the practice slide.

One call to compute_metrics() per slide returns ONE summary row, so slides
and groups can be stacked into a single table for statistics.

Design rules for cross-slide comparability
------------------------------------------
- The pruning cutoff is passed IN and must be the SAME for every slide.
  It is never re-estimated per slide. Choose it once, e.g. with
  tukey_cutoff() on edge lengths POOLED across slides, or from a biological
  rationale, and record it in docs/validation_log.md.
- Distances are reported in px, and also in um when um_per_px is given.
  Scanner calibration is required before cross-slide comparison.
- Scalar metrics are normalized for cell number and graft size where
  possible (Clark-Evans R, fractions, enrichment ratios).
- Topology metrics (degree, clustering) are largely set by the triangulation
  itself; compare them only between slides processed identically.

Edge construction
-----------------
- "delaunay": edges of the Delaunay triangulation (scipy), the classical
  cell-graph definition; parameter-free. Same edges as
  squidpy.gr.spatial_neighbors(delaunay=True).
- "knn": each cell linked to its k nearest cells, symmetrized (edge kept if
  either cell chose the other), as in squidpy with n_neighs=k.

Coordinates are (y, x) = (row, col) in pixels, matching the annotation CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree


# --- Graph construction ------------------------------------------------------

def build_edges(points_yx, method="delaunay", k=6):
    """Undirected edge table (i < j) with Euclidean length in px."""
    xy = np.asarray(points_yx, dtype=float)[:, ::-1]
    if method == "delaunay":
        s = Delaunay(xy).simplices
        pairs = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    elif method == "knn":
        _, idx = cKDTree(xy).query(xy, k=k + 1)      # first hit is the cell itself
        pairs = np.column_stack([np.repeat(np.arange(len(xy)), k),
                                 idx[:, 1:].ravel()])
    else:
        raise ValueError(f"method must be 'delaunay' or 'knn', got {method!r}")

    pairs = np.unique(np.sort(pairs, axis=1), axis=0)  # undirected, no duplicates
    d = xy[pairs[:, 0]] - xy[pairs[:, 1]]
    return pd.DataFrame({"i": pairs[:, 0], "j": pairs[:, 1],
                         "length_px": np.hypot(d[:, 0], d[:, 1])})


def tukey_cutoff(lengths, k=1.5):
    """Tukey outlier fence Q3 + k*IQR. Apply to POOLED lengths across slides."""
    q1, q3 = np.quantile(np.asarray(lengths, dtype=float), [0.25, 0.75])
    return q3 + k * (q3 - q1)


def prune_edges(edges, cutoff_px):
    """Drop edges longer than cutoff_px."""
    return edges[edges["length_px"] <= cutoff_px].reset_index(drop=True)


def to_networkx(edges, n_nodes):
    """NetworkX graph with every cell as a node (isolated cells kept)."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_weighted_edges_from(zip(edges["i"], edges["j"], edges["length_px"]),
                              weight="length_px")
    return G


# --- Metrics -------------------------------------------------------------------

def compute_metrics(points_yx, graft_mask, cutoff_px, *, method="delaunay", k=6,
                    um_per_px=None, n_radial_bins=10, core_depth=0.5,
                    sample_id=None):
    """Build, prune and measure one slide's cell graph.

    Parameters
    ----------
    points_yx : (n, 2) array of cell centers, (row, col) in px.
    graft_mask : 2D bool array, True inside the graft (same frame as points).
    cutoff_px : fixed edge-length cutoff, identical across slides.
    um_per_px : scanner calibration; adds um / mm^2 versions when given.
    core_depth : cells deeper than this fraction of the maximum depth
        (distance to the graft boundary) count as "core" for core_enrichment.

    Returns
    -------
    summary : dict, one row of scalar metrics plus the settings used
    nodes : DataFrame, per-cell metrics
    radial : DataFrame, radial density profile from the graft centroid
    edges : DataFrame, pruned edge table
    """
    pts = np.asarray(points_yx, dtype=float)
    mask = np.asarray(graft_mask, dtype=bool)
    n = len(pts)
    xy = pts[:, ::-1]

    edges_all = build_edges(pts, method=method, k=k)
    edges = prune_edges(edges_all, cutoff_px)
    G = to_networkx(edges, n)

    # Graft geometry: centroid and area from the MASK, not the cells
    cy, cx = ndimage.center_of_mass(mask)
    area_px = float(mask.sum())
    equiv_radius_px = np.sqrt(area_px / np.pi)

    # Nearest-neighbor distance + Clark-Evans (no edge correction)
    nnd = cKDTree(xy).query(xy, k=2)[0][:, 1]
    expected_nnd = 0.5 / np.sqrt(n / area_px)
    clark_evans_R = nnd.mean() / expected_nnd
    se = 0.26136 / np.sqrt(n ** 2 / area_px)
    clark_evans_z = (nnd.mean() - expected_nnd) / se

    # Components
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    comp_id = {v: c for c, comp in enumerate(comps) for v in comp}

    # Shape-aware depth: distance to graft boundary, normalized 0 (edge) -> 1 (deepest)
    depth = ndimage.distance_transform_edt(mask)
    depth_norm = depth / depth.max()
    rows = np.clip(np.round(pts[:, 0]).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.round(pts[:, 1]).astype(int), 0, mask.shape[1] - 1)
    cell_depth = depth_norm[rows, cols]
    core_area_frac = (depth_norm[mask] >= core_depth).mean()
    core_cell_frac = (cell_depth >= core_depth).mean()
    core_enrichment = core_cell_frac / core_area_frac   # <1 = depleted core

    # Orientation-free asymmetry: how far the cells' center of mass sits from
    # the graft centroid, relative to graft size (0 = centered)
    dy, dx = pts[:, 0].mean() - cy, pts[:, 1].mean() - cx
    centroid_offset_norm = np.hypot(dx, dy) / equiv_radius_px
    centroid_offset_angle_deg = np.degrees(np.arctan2(-dy, dx))  # 0=right, 90=up

    # Per-cell table
    clust = nx.clustering(G)
    nodes = pd.DataFrame({
        "cell": np.arange(n),
        "y": pts[:, 0],
        "x": pts[:, 1],
        "degree": [G.degree(v) for v in range(n)],
        "clustering": [clust[v] for v in range(n)],
        "mean_edge_len_px": [np.mean([d["length_px"] for _, _, d in G.edges(v, data=True)])
                             if G.degree(v) else np.nan for v in range(n)],
        "nnd_px": nnd,
        "dist_center_px": np.hypot(xy[:, 0] - cx, xy[:, 1] - cy),
        "depth_norm": cell_depth,
        "component": [comp_id[v] for v in range(n)],
    })

    # Radial profile from the graft centroid; ring areas measured on the mask
    yy, xx = np.nonzero(mask)
    r_mask = np.hypot(xx - cx, yy - cy)
    r_bins = np.linspace(0, r_mask.max(), n_radial_bins + 1)
    ring_area, _ = np.histogram(r_mask, bins=r_bins)
    ring_cells, _ = np.histogram(nodes["dist_center_px"], bins=r_bins)
    mean_density = n / area_px * 1e5
    ring_density = ring_cells / np.maximum(ring_area, 1) * 1e5
    radial = pd.DataFrame({
        "r_inner_px": r_bins[:-1],
        "r_outer_px": r_bins[1:],
        "r_mid_frac": (r_bins[:-1] + r_bins[1:]) / 2 / r_bins[-1],
        "n_cells": ring_cells,
        "ring_area_px": ring_area,
        "cells_per_1e5_px2": ring_density,
        "density_rel_to_mean": ring_density / mean_density,
    })

    deg = nodes["degree"].to_numpy()
    summary = {
        # settings
        "sample_id": sample_id,
        "method": method,
        "k": k if method == "knn" else None,
        "cutoff_px": cutoff_px,
        "um_per_px": um_per_px,
        # size
        "n_cells": n,
        "graft_area_px": area_px,
        "n_edges_before_prune": len(edges_all),
        "n_edges": G.number_of_edges(),
        # topology (compare only between identically processed slides)
        "mean_degree": deg.mean(),
        "mean_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "n_components": len(comps),
        "largest_component": len(comps[0]),
        "frac_largest_component": len(comps[0]) / n,
        "n_isolated": int((deg == 0).sum()),
        # spacing / point pattern
        "mean_nnd_px": nnd.mean(),
        "expected_nnd_csr_px": expected_nnd,
        "clark_evans_R": clark_evans_R,
        "clark_evans_z": clark_evans_z,
        "cells_per_1e5_px2": mean_density,
        # spatial distribution within the graft
        "core_enrichment": core_enrichment,
        "centroid_offset_norm": centroid_offset_norm,
        "centroid_offset_angle_deg": centroid_offset_angle_deg,
    }
    if um_per_px is not None:
        summary.update({
            "cutoff_um": cutoff_px * um_per_px,
            "mean_nnd_um": nnd.mean() * um_per_px,
            "cells_per_mm2": n / (area_px * um_per_px ** 2) * 1e6,
        })

    return summary, nodes, radial, edges