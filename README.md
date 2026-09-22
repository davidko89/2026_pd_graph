# 2026_PD_Graph: Spatial Cell-Graph Analysis of PD Graft Integration

**Author**: Chanyoung Ko
**Date**: 2026-09-22
**Last Updated**: 2026-09-22

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
   within graft and host tissue regions (via CellViT++, cross-validated against
   CNS tissue given its cancer/tumor-heavy training data).
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
| **torch_geometric** | TBD | Graph neural networks |
| **trident** | TBD | WSI segmentation / patching / feature extraction |
| **cellvit-plus-plus** | TBD | Cell segmentation & classification |
| **scanpy / anndata / squidpy** | TBD | Spatial transcriptomics data handling |
| **networkx** | TBD | Graph construction / analysis |

*(Full pinned versions to be recorded via `conda env export > docs/environment_pathology_gpu.yml`
once installation is complete.)*

---

## Project Structure

See `docs/` for the full project/data folder convention (mirrors the
`2026_asd_immune` template: code/results in `projects/`, raw/intermediate
data in a same-named sibling `data/` folder, split by mutability).

---

## Contact

**David Chanyoung Ko, MD-PhD**
Kim Laboratory
