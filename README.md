# CytoBridge Downstream

This repository contains dataset-specific downstream analysis notebooks built on top of the main `CytoBridge` package. The downstream code is organized as thin dataset wrappers around `CytoBridge` APIs rather than a separate implementation stack.

## Usage

Install `CytoBridge` first from the main `cytobridge-spatial` repository:

```bash
git clone https://github.com/zhenyiizhang/cytobridge-spatial.git
cd cytobridge-spatial

conda env create -f environment.yml
conda activate cb_pipeline
pip install -e .
```

Then clone this repository and open the notebooks here in the same `cb_pipeline` environment.

Most notebooks use paths relative to this repository. The MOSTA support files used by the notebooks are under `assets/mosta/`.

The canonical ARISTA workflow loads the trained model, edge classifier, and annotated model-input CSV under `assets/arista/` and `data/arista/` through the installed `CytoBridge` API. It does not import `vendor/legacy_arista_stack`. Older parity notebooks remain temporarily available as migration references while their figures are ported one by one. The few large attention arrays are stored as release files; see "Large reviewer assets" below.

The AD mouse notebooks use the runtime files under `assets/admouse/` together with the AD mouse inputs under `data/admouse/`.

The zebrafish notebook uses the runtime files under `assets/zebrafish_runtime/`.

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
- `data/arista/`: ARISTA 52-dimensional annotated model-input CSV used for checkpoint inference
- `data/admouse/`: local AD mouse inputs expected by the AD mouse notebooks

## Data

The processed files under `data/` are included here when they are small enough for Git. A copy of the `data/` directory is also available on Google Drive:

https://drive.google.com/drive/folders/1L_TjHU4TsbY8dc6QJlhWqnKaQl6QWBfV?usp=sharing

The larger files needed by the notebooks are listed in the next section.

## Large reviewer assets

GitHub blocks regular Git files larger than 100MB. For that reason, a few large reviewer files are attached to the `reviewer-assets-v1` GitHub Release instead of being committed to the repository.

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

- `notebooks/arista/arista_velocity_t1_streams_api.ipynb`
  Canonical package-backed ARISTA example. It loads the trained checkpoint with
  `CytoBridge.tl.load_legacy_dynamical_model_from_dir`, computes the velocity
  decomposition with `CytoBridge.tl.compute_velocity_components`, and renders
  the six t1 panels with `CytoBridge.pl.plot_velocity_component`. The notebook
  fails if the external vendored `DeepRUOT` runtime is imported.

- `notebooks/mosta/mosta_interpolated_slices.ipynb`
  Runs MOSTA with piecewise spatial warp and displays the interpolated slice snapshots.

- `notebooks/mosta/mosta_lineage_sankey_3d.ipynb`
  Runs MOSTA without piecewise spatial warp and displays the lineage Sankey plot together with the 3D spatiotemporal output.

- `notebooks/mosta/mosta_virtual_video.ipynb`
  Runs the complete MOSTA baseline video workflow from model output generation to frame export and GIF assembly.

- `notebooks/mosta/mosta_telencephalon_velocity_stream_grid.ipynb`
  Reproduces the telencephalon velocity stream/grid plots from the MOSTA files in `assets/mosta/`.

- `notebooks/mosta/mosta_velocity_communication_focus_brain_t3.ipynb`
  Reproduces the MOSTA velocity communication focus plots at brain timepoint 3.

- `notebooks/mosta/mosta_wnt3a_fzd7_lrp6_total_hotspot_triptych.ipynb`
  Reproduces the Wnt3a-Fzd7-Lrp6 total hotspot triptych figure from local copied plotting inputs.

- `notebooks/arista/arista_lineage_snapshot_focus_anchor.ipynb`
  Legacy migration reference; not part of the canonical package-backed workflow yet.

- `notebooks/arista/arista_spatiotemporal_3d_focus_anchor.ipynb`
  Legacy migration reference; not part of the canonical package-backed workflow yet.

- `notebooks/arista/arista_growth_interaction_celltype_bubble.ipynb`
  Legacy migration reference; not part of the canonical package-backed workflow yet.

- `notebooks/arista/arista_velocity_spatial_direction_correlation_roi_t1_scvelo_only.ipynb`
  Legacy migration reference; not part of the canonical package-backed workflow yet.

- `notebooks/arista/arista_velocity_t1_streams.ipynb`
  Archived legacy implementation superseded by `arista_velocity_t1_streams_api.ipynb`.

- `notebooks/admouse/admouse_gene_expression.ipynb`
  Reproduces the AD mouse gene-expression view from the local AD mouse runtime and plotting inputs.

- `notebooks/admouse/admouse_lr_score.ipynb`
  Reproduces the AD mouse ligand-receptor score figure from the local AD mouse preprocessing outputs.

- `notebooks/admouse/admouse_spatial_comparison.ipynb`
  Reproduces the AD mouse spatial comparison panels from the local interpolated AD mouse slices.

- `notebooks/zebrafish/zebrafish_api_subfigures.ipynb`
  Generates zebrafish downstream subfigures through package-backed APIs and the files in `assets/zebrafish_runtime/`.

## Reviewer reproduction

### MOSTA

1. Install the main `cb_pipeline` repository and activate the `cb_pipeline` environment.
2. Clone this repository.
3. Run `bash scripts/download_reviewer_assets.sh`.
4. Open any notebook under `notebooks/mosta/`.
5. Run all cells.

### ARISTA

1. Install the main `cytobridge-spatial` repository and activate its isolated environment.
2. Clone this repository.
3. Open `notebooks/arista/arista_velocity_t1_streams_api.ipynb`.
4. Run all cells. For a quick smoke test, set `CYTOBRIDGE_MAX_CELLS=256` before starting Jupyter; leave it unset for the complete t1 slice.

The same workflow is available as a non-interactive command. With the portable
assets committed to this repository, no workspace override is required:

```bash
python scripts/arista_velocity_t1_streams_api.py \
  --device cuda \
  --output-dir results/arista_velocity_t1_streams_api
```

To use the original project workspace on the shared server instead, add
`--workspace-root /data/cytobridge/projects/CytoBridge-ST-1104/workspace`.
The run manifest records resolved paths, SHA-256 hashes, the installed
`CytoBridge` module, model stages, and the velocity-decomposition identity
check.

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
