# 2026_PD_Graph: Spatial Cell-Graph Analysis of PD Graft Integration

**Author**: Chanyoung Ko
**Date**: 2026-09-22
**Last Updated**: 2026-09-23

---

## Overview

This repository contains the analysis pipeline for characterizing **iPSC-derived
dopaminergic graft integration** in a Parkinson's disease model, using digital
pathology (H&E/IHC whole-slide imaging) and graph-based spatial analysis of
cells and their local neighborhoods (including synaptic structures).

The project proceeds in phases:

1. **Whole-slide image preprocessing**: Tissue segmentation, patch extraction,
   and foundation-model feature extraction from H&E/IHC slides (via Trident).
2. **Cell segmentation & classification**: Cell-level detection and typing
   within graft and host tissue regions (via CellViT-Inference, cross-validated
   against CNS tissue given its cancer/tumor-heavy training data).
3. **Spatial/graph analysis**: Construction of cell-neighborhood graphs and
   graph neural network modeling (via PyTorch Geometric / NetworkX) to relate
   spatial organization to graft integration outcomes.
4. **Public data overlay**: Where own spatial transcriptomic data isn't
   available, mapping to public spatial atlases (SEA-AD, HEST, 10x Xenium
   mouse brain) via registration (ABBA/DeepSlice) or label transfer
   (Tangram/CellTrek), to provide healthy-control spatial context.

---

## Software Versions

### Python Environment (`pathology_gpu` conda env, based on proven `cv_gpu` pins)
- **Python**: 3.10.14

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | 2.9.1+cu130 | Deep learning backend (RTX 5080 / Blackwell, requires cu128+) |
| **torchvision** | 0.24.1+cu130 | Vision model support |
| **torch_geometric** | 2.8.0.post1 | Graph neural networks |
| **pyg_lib** | 0.7.0+pt29cu130 | PyG accelerated ops |
| **torch_scatter** | 2.1.2+pt29cu130 | PyG accelerated ops |
| **torch_sparse** | 0.6.18+pt29cu130 | PyG accelerated ops |
| **trident** | commit `e4c98b2` (editable install) | WSI segmentation / patching / feature extraction |
| **cellvit** | 1.0.9 | Cell segmentation & classification (CellViT-Inference) |
| **networkx** | 3.4.2 | Graph construction / analysis |
| **scanpy** | 1.11.5 | Spatial/single-cell transcriptomics |
| **anndata** | 0.11.4 | Single-cell data structures |
| **squidpy** | 1.6.5 | Spatial-omics analysis, spatial neighbor graphs |

*(Full pinned versions, including transitive dependencies, recorded in `docs/requirements_pathology_gpu_full.txt`.)*

---

## Project Structure

See `docs/` for the full project/data folder convention (mirrors the
`2026_asd_immune` template: code/results in `projects/`, raw/intermediate
data in a same-named sibling `data/` folder, split by mutability).

---

## Contact

**David Chanyoung Ko, MD-PhD**
Kim Laboratory
