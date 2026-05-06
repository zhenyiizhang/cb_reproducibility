# CytoBridge Downstream

This repository contains dataset-specific downstream analysis notebooks built on top of the main `CytoBridge` package. The downstream code is organized as thin dataset wrappers around `CytoBridge` APIs rather than a separate implementation stack.

## Usage

Install `CytoBridge` first by following the installation steps in the main `cb_pipeline` repository:

```bash
git clone <cb_pipeline-repo>
cd cb_pipeline

conda env create -f environment.yml
conda activate cb_pipeline
pip install -e .
```

Then clone this repository and open the notebooks here in the same `cb_pipeline` environment.

For the curated MOSTA notebooks listed below, no local path edits, external result directories, or manual cache setup are required. The required MOSTA assets are bundled in this repository under `assets/mosta/`.

The ARISTA notebooks are also self-contained after downloading the reviewer assets. The repository ships the legacy ARISTA model, edge classifier, annotation CSV, and classifier cache under `assets/arista/`, `data/arista/`, and `vendor/legacy_arista_stack/`, so reviewers can run the notebooks directly without pointing to an external Desktop checkout.

The AD mouse notebooks use the runtime assets bundled under `assets/admouse/` together with the AD mouse inputs under `data/admouse/`.

The zebrafish notebook is also self-contained. The repository ships a minimal zebrafish runtime bundle under `assets/zebrafish_runtime/`, so reviewers can run the notebook directly without preparing external data paths.

## Layout

- `notebooks/mosta/`: MOSTA downstream analysis notebooks
- `notebooks/arista/`: ARISTA downstream notebooks
- `notebooks/admouse/`: AD mouse downstream notebooks
- `notebooks/zebrafish/`: zebrafish downstream notebooks
- `downstream_helpers/`: thin dataset-specific wrappers that assemble inputs, parameters, and outputs around `CytoBridge` APIs
- `scripts/`: thin CLI entrypoints built on top of `downstream_helpers/`
- `assets/mosta/`: MOSTA local runtime assets copied into this repository
- `assets/arista/`: ARISTA local runtime assets, classifier cache, and plotting metadata
- `assets/admouse/`: AD mouse runtime assets, plotting configuration, and pretrained model files
- `assets/zebrafish/`: small zebrafish metadata assets needed by the public wrappers
- `assets/zebrafish_runtime/`: minimal zebrafish runtime bundle used by the zebrafish notebook and CLI
- `data/mosta/`: MOSTA input CSV used by the reproduction runner
- `data/arista/`: ARISTA input CSV used by the legacy reviewer notebooks
- `data/admouse/`: local AD mouse inputs expected by the AD mouse notebooks

## Large reviewer assets

GitHub rejects regular Git files larger than 100MB. The large reviewer inputs are therefore published as assets on the `reviewer-assets-v1` GitHub Release, while this repository keeps the code, notebooks, and smaller cached outputs in Git.

After cloning the repository, restore the large files to their expected paths:

```bash
bash scripts/download_reviewer_assets.sh
```

The script downloads:

- `data/mosta/mosta_four_time_with_celltype_refined.csv`
- `results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t0.0.npy`
- `results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t0.5.npy`
- `results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t1.0.npy`
- `results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t1.5.npy`
- `results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t2.0.npy`

## Current notebooks

- `notebooks/mosta/mosta_interpolated_slices.ipynb`
  Runs MOSTA with piecewise spatial warp and displays the interpolated slice snapshots.

- `notebooks/mosta/mosta_lineage_sankey_3d.ipynb`
  Runs MOSTA without piecewise spatial warp and displays the lineage Sankey plot together with the 3D spatiotemporal output.

- `notebooks/mosta/mosta_virtual_video.ipynb`
  Runs the complete MOSTA baseline video workflow from model output generation to frame export and GIF assembly.

- `notebooks/mosta/mosta_telencephalon_velocity_stream_grid.ipynb`
  Reproduces the telencephalon velocity stream/grid plots from the bundled MOSTA assets.

- `notebooks/mosta/mosta_velocity_communication_focus_brain_t3.ipynb`
  Reproduces the MOSTA velocity communication focus plots at brain timepoint 3.

- `notebooks/mosta/mosta_wnt3a_fzd7_lrp6_total_hotspot_triptych.ipynb`
  Reproduces the Wnt3a-Fzd7-Lrp6 total hotspot triptych figure from local copied plotting inputs.

- `notebooks/arista/arista_lineage_snapshot_focus_anchor.ipynb`
  Runs the ARISTA legacy focus-anchor lineage pipeline and displays the lineage Sankey together with the timepoint snapshots.

- `notebooks/arista/arista_spatiotemporal_3d_focus_anchor.ipynb`
  Runs the ARISTA legacy focus-anchor 3D communication pipeline and displays reviewer-facing static exports.

- `notebooks/arista/arista_growth_interaction_celltype_bubble.ipynb`
  Reproduces the ARISTA growth-versus-interaction cell-type bubble plot through the legacy runtime.

- `notebooks/arista/arista_velocity_spatial_direction_correlation_roi_t1_scvelo_only.ipynb`
  Reproduces the ARISTA ROI cosine-correlation panel at timepoint 1 using scVelo smoothing only.

- `notebooks/arista/arista_velocity_t1_streams.ipynb`
  Reproduces the ARISTA `scvelo_streams` family at timepoint 1 for intrinsic, interaction, and full velocity in spatial and gene views.

- `notebooks/admouse/admouse_gene_expression.ipynb`
  Reproduces the AD mouse gene-expression view from the local AD mouse runtime and plotting inputs.

- `notebooks/admouse/admouse_lr_score.ipynb`
  Reproduces the AD mouse ligand-receptor score figure from the local AD mouse preprocessing outputs.

- `notebooks/admouse/admouse_spatial_comparison.ipynb`
  Reproduces the AD mouse spatial comparison panels from the local interpolated AD mouse slices.

- `notebooks/zebrafish/zebrafish_api_subfigures.ipynb`
  Generates zebrafish downstream subfigures through package-backed APIs and the bundled zebrafish runtime.

## Reviewer reproduction

### MOSTA

1. Install the main `cb_pipeline` repository and activate the `cb_pipeline` environment.
2. Clone this repository.
3. Run `bash scripts/download_reviewer_assets.sh`.
4. Open any curated notebook under `notebooks/mosta/`.
5. Run all cells.

### ARISTA

1. Install the main `cb_pipeline` repository and activate the `cb_pipeline` environment.
2. Clone this repository.
3. Run `bash scripts/download_reviewer_assets.sh`.
4. Open any notebook under `notebooks/arista/`.
5. Run all cells.

### AD Mouse

1. Install the main `cb_pipeline` repository and activate the `cb_pipeline` environment.
2. Clone this repository.
3. Open any notebook under `notebooks/admouse/`.
4. Run all cells.

### Zebrafish

1. Install the main `cb_pipeline` repository and activate the `cb_pipeline` environment.
2. Clone this repository.
3. Open `notebooks/zebrafish/zebrafish_api_subfigures.ipynb`.
4. Run all cells.

The notebook writes the generated subfigures under `results/`.
