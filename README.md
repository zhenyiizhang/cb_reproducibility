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

The panel-by-panel scope, formal evaluation contract, and migration status for
main Figure 5 and Supplementary Figures S12-S17 are tracked in
[`docs/arista_reproduction_matrix.md`](docs/arista_reproduction_matrix.md).

For a newly retrained model, the same notebooks accept an aligned H5AD and a
current checkpoint directory through `ARISTA_ALIGNED_H5AD`,
`ARISTA_MODEL_DIR`, and `ARISTA_MODEL_FORMAT=current`. Dataset-specific code in
this repository only resolves inputs, timepoints, and styling; preprocessing,
training, simulation, classifier caching, interaction, lineage, and plotting
remain in the installed `CytoBridge` package.

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

- `notebooks/arista/arista_lineage_snapshot_api.ipynb`
  Runs package-backed piecewise interpolation with KNN=1, regenerates paired
  observed/generated spatial snapshots, and exports the lineage Sankey. It can
  use either the portable published checkpoint or a current retrained model.

- `notebooks/arista/arista_spatiotemporal_3d_api.ipynb`
  Extends the same package-backed workflow with attention-based interactions
  and the reaEGC focus-anchor 3D panel. The notebook regenerates attention
  arrays and does not load precomputed interaction matrices.

- `notebooks/arista/arista_growth_interaction_api.ipynb`
  Recomputes growth and interaction magnitude through the shared component API,
  aggregates by time/cell type, and redraws the bubble panel.

- `notebooks/arista/arista_velocity_direction_correlation_api.ipynb`
  Recomputes the full t1 spatial and PC1-PC2 velocity streams plus the
  full-versus-interaction spatial direction cosine in a reaEGC-defined ROI
  using the shared scVelo projection API.

- `notebooks/arista/arista_temporal_gene_lr_patterns_api.ipynb`
  Recomputes dense-time gene programs and expression-aware ligand-receptor
  trajectories for Supplementary Figures S15-S17. PCA inversion, temporal
  clustering, offline GMT enrichment, LR projection, and plotting are package
  APIs; the notebook only supplies the ARISTA reference H5AD, LR database,
  gene-set GMT, time grid, and cluster count.
  Its prospective default prefers human-tagged symbols (89 LR pairs on these
  inputs); the explicit paper-parity mode loads archived PCA sidecars and uses
  the historical first-symbol rule (68 pairs).

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
  Archived legacy migration reference superseded by `arista_lineage_snapshot_api.ipynb`.

- `notebooks/arista/arista_spatiotemporal_3d_focus_anchor.ipynb`
  Archived legacy migration reference superseded by `arista_spatiotemporal_3d_api.ipynb`.

- `notebooks/arista/arista_growth_interaction_celltype_bubble.ipynb`
  Archived legacy migration reference superseded by `arista_growth_interaction_api.ipynb`.

- `notebooks/arista/arista_velocity_spatial_direction_correlation_roi_t1_scvelo_only.ipynb`
  Archived legacy migration reference superseded by `arista_velocity_direction_correlation_api.ipynb`.

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
3. Open one of the `*_api.ipynb` notebooks under `notebooks/arista/`.
4. Run all cells. For a quick smoke test, set `CYTOBRIDGE_MAX_CELLS=256` before starting Jupyter; leave it unset for the complete t1 slice.

For the lineage/3D notebooks, set `CYTOBRIDGE_SMOKE=1` for a 16-particle
wiring test. Leave it unset for the 7,668-particle reproduction. To compare a
new full run with the published checkpoint, launch the notebooks twice:

```bash
# Published portable checkpoint (default)
unset ARISTA_MODEL_FORMAT ARISTA_ALIGNED_H5AD ARISTA_MODEL_DIR

# Newly retrained current checkpoint
export ARISTA_MODEL_FORMAT=current
export ARISTA_ALIGNED_H5AD=/path/to/arista-full/preprocess/arista_aligned.h5ad
export ARISTA_MODEL_DIR=/path/to/arista-full/training
```

Both modes execute `CytoBridge.tl.run_interpolation_workflow`,
`compute_timepoint_communications`, `plot_lineage_sankey`, and
`plot_spatiotemporal_3d`; the compatibility loader changes only how the saved
published model is instantiated.

For the formal current-model Figure 5a/5b and S13/S14 rerun, use the single
no-warp command below. It fits a fresh classifier by default and refuses a
non-empty cache directory unless reuse is requested explicitly:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_arista_formal_panels_api.py \
  --aligned-h5ad /path/to/arista-full/preprocess/arista_aligned.h5ad \
  --model-dir /path/to/arista-full/training \
  --output-dir /path/to/new-review-folder/formal-no-warp \
  --device cuda \
  --random-seed 42 \
  --classifier-knn-neighbors 1
```

The defaults are the full formal contract: current checkpoint, observed model
times `0,1,2,3,4`, interpolated times `0.5,1.5,2.5,3.5`, 7,668 particles,
non-split `dt=0.05`, split `dt=0.01`, split `sigma=0.03`, no piecewise spatial
warp, and recomputed communication plus the 3D panel. Non-split fixed particles
are used for lineage ribbons/Sankey; split-SDE populations are used for
generated clouds and communication. With `k=1`, KNN refinement leaves each raw
classifier prediction unchanged. The output directory contains
`spatiotemporal_3d.*` (Figure 5a), `snapshots/time_0.5.*` (Figure 5b),
`growth_dense_time_grid.*` (S13), `lineage_sankey.*` (S14a),
`celltype_composition.*` (S14b), the attention/communication artifacts, the
classifier cache, `run_manifest.json`, and `formal_panel_index.json`.

The prospective S15-S17 workflow requires a package-processed reference H5AD
that retains `varm['PCs']` and the ligand-receptor database:

```bash
export ARISTA_REFERENCE_H5AD=/path/to/arista-full/preprocess/arista_aligned.h5ad
export ARISTA_LR_DATABASE=/path/to/CellChatDB.ligrec.human.csv
export ARISTA_GENE_SET_GMT=/path/to/GO_Biological_Process_2023.gmt
```

The older source H5AD contains PCA coordinates but not the complete inverse-PCA
contract. For a declared historical reproduction, provide the archived loading
and center tables explicitly; the generic loader validates feature alignment
and records both file hashes:

```bash
export ARISTA_PAPER_PARITY=1
export ARISTA_PCA_COMPONENTS_CSV=/path/to/arista_pca_components_with_gene_names.csv
export ARISTA_PCA_CENTER_CSV=/path/to/arista_pca_mean.csv
# Optional: reuse a classifier only when its metadata match the requested run.
export ARISTA_CLASSIFIER_CACHE_PATH=/path/to/classifier_resmlp.pt
```

Paper-parity mode uses 3,072 particles, KNN=10, the split-SDE-only legacy
simulation contract, first-symbol ligand/receptor mapping, Ward linkage, and
dendrogram order. It also uses the complete GMT library as the S15 enrichment
background, mirroring the archived table's organism-wide rather than
expression-only contract; prospective mode uses the reconstructed expression
genes as the explicit background. The exact universe remains database-version
dependent: the fixed Enrichr GO BP 2023 GMT has 14,698 genes, whereas the
archived clusterProfiler BP table records 18,870. It reproduces the historical
68-pair input set without embedding ARISTA-specific parsing in the package.

For a quantitative saved-versus-retrained comparison, first complete the
package full run, then evaluate the published checkpoint with exactly the same
particle count, integration step, OT cap, and seed:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_arista_model_comparison.py \
  --current-metrics /path/to/arista-full/downstream/distribution_evaluation/distribution_metrics.csv \
  --current-name recovered_six_stage_fixed_rbf_current_preprocess_auto_thresholds \
  --current-aligned-h5ad /path/to/arista-full/preprocess/arista_aligned.h5ad \
  --current-threshold-meta /path/to/arista-full/preprocess/edge_classifier/arista_edge_model.pt.meta.json \
  --additional-metrics recovered_six_stage_fixed_rbf_legacy_input=/path/to/legacy-input-run/downstream/distribution_evaluation/distribution_metrics.csv \
  --additional-metrics recovered_six_stage_fixed_rbf_strict_frozen_auto_edge_published_thresholds=/path/to/strict-paired-run/downstream/distribution_evaluation/distribution_metrics.csv \
  --output-dir results/arista_published_vs_retrained \
  --device cuda \
  --n-samples 5000 \
  --dt 0.01 \
  --interaction-m 1024 \
  --max-ot-points 1024 \
  --structure-max-points 5000 \
  --random-seed 42
```

The script calls only public package evaluation/comparison APIs. It writes the
old-model generated-versus-observed PCA/spatial panels, paired W1/W2/TMV and
local-structure tables and deltas, a time-resolved comparison figure,
subsampled distribution arrays, and
`coordinate_threshold_comparison.json`. The latter checks coordinate ranges,
nearest-neighbor scale, and `spatial_cutoff / median_nn` before interpreting
the historical `0.05/0.45` thresholds against full-preprocess auto thresholds.
Pass `--legacy-metrics /path/to/old/distribution_metrics.csv` to reuse a
completed old-model evaluation, and repeat
`--additional-metrics NAME=/path/to/distribution_metrics.csv` for matched
threshold/input controls. The formal ARISTA audit uses four stable conditions:
the published saved model; recovered six-stage fixed-RBF training on the exact
legacy model input; recovered six-stage fixed-RBF training on current
preprocessing with automatically selected thresholds; and a strict paired
control that freezes that auto condition's H5AD and edge weights while changing
only the fit-time thresholds to `0.05/0.45`. A trainable-RBF legacy-input run is
retained only as an explicitly labelled implementation ablation and is excluded
from the formal retraining comparison. Independently preprocessing with a
different threshold policy can also change the fitted edge classifier, so it is
not a threshold-only control; only the frozen-edge pair supports that causal
interpretation. The package-level
`convert_legacy_model_input_csv.py` command maps
the released table's `x1,x2` to spatial coordinates and `x3..x52` to latent
dimensions while recording that gene expression is unavailable. See
`docs/arista_reproduction_matrix.md` for the causal interpretation of each
pairwise comparison.

The completed full-data comparison selected the current-preprocess/automatic-
threshold retrain. Mean spatial metrics across the five observed stages were:

| condition | W1 | W2 | TMV | NN dispersion | clump fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| published saved model | 0.07377 | 0.08815 | 0.02321 | 0.9449 | 0.0110 |
| recovered six-stage, legacy input | 0.06308 | 0.07515 | 0.01366 | 0.7981 | 0.0261 |
| recovered six-stage, current preprocess + automatic thresholds | 0.06648 | 0.08182 | 0.00624 | 0.8966 | 0.0127 |
| same current input/edge weights, fit with `0.05/0.45` | 0.06949 | 0.08095 | 0.00652 | 0.3128 | 0.0976 |

The strict `0.05/0.45` control collapses into repeated local spatial clumps
despite competitive W1/W2, so W1/W2/TMV alone are not sufficient for model
selection. The historical/current coordinate scales are nearly identical;
the different spatial cutoffs are policy choices, not a consequence of
coordinate rescaling.

For S16-S17, the paper-parity published and selected-retrained runs share the
same 68 LR pairs. Their continuous aggregate scores are strongly correlated
(Pearson 0.966; Spearman 0.978), while the two-cluster partition is unstable:
published recomputation 24/44, selected retrain 11/57, ARI -0.048. The frozen
manuscript table remains exactly 22/46; 66 of 68 published-recomputation labels
match it. This is reported as a clustering-boundary sensitivity rather than
silently forcing the historical labels.

Use `scripts/compare_temporal_lr_runs.py` to compare any two completed temporal
runs. It writes score correlations, pairwise trajectory correlations/min-max
RMSE, cluster agreement/ARI/NMI, merged tables, and a comparison figure.

Use `scripts/enrich_arista_temporal_gene_patterns.py` to add full 2,000-gene
offline GMT enrichment to an existing package-generated temporal run. The
script records the GMT hash and background contract. For an exact visual redraw
of the archived S15 GO tables, use
`scripts/redraw_arista_frozen_s15_enrichment.py`; it only adapts the historical
clusterProfiler columns, then calls the public CytoBridge plotting API.

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
