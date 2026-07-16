# ARISTA manuscript reproduction matrix

This document maps the ARISTA/axolotl claims and panels in
`cytobridge_manuscript_latest_clean.docx` and `si.pdf` to package-backed
reproduction entry points. The notebooks in this repository are deliberately
thin dataset adapters: model loading, simulation, interaction calculation,
classification, lineage construction, evaluation, and plotting live in the
installed `CytoBridge` package.

## Evaluation contract

Final quantitative comparisons must use the same simulation and optimal
transport settings for every model:

- full aligned dataset (46,189 cells; 2 spatial + 50 PCA dimensions);
- 5,000 simulated particles, `dt=0.01`, `sigma=0.03`;
- score and interaction components enabled, `interaction_m=1024`;
- exact OT capped at 1,024 points per empirical distribution;
- random seed 42;
- W1 and W2 in joint, spatial, and PCA spaces;
- TMV as relative total-mass variation, matching the historical notebook.
- nearest-neighbor dispersion, support precision/recall, and clump fraction,
  because W1/W2 alone can miss locally collapsed particle clouds;
- saved generated/observed samples so every metric and panel can be audited.

Smoke runs are wiring tests only. Their cell count, particle count, classifier
fit, graph density, and thresholds are not scientific results and must not be
used in the manuscript comparison.

Two comparison layers are kept separate.

The threshold/stability ablation uses the compact three-stage
`500/2001/1000` profile (`arista_spatial_compact_full.yaml`) for both retrained
conditions:

1. `published_saved_model`: portable checkpoint with the historical
   spatial/edge thresholds `0.05/0.45`;
2. `compact_retrained_auto_thresholds`: current end-to-end preprocessing and training,
   with thresholds selected from the full data;
3. `compact_retrained_published_thresholds_stabilized`: current training code on the
   exact same aligned/PCA input as condition 2, using `0.05/0.45` and explicit
   finite-loss/gradient safeguards.

Conditions 2 and 3 must share identical `spatial_aligned`, `X_latent`, cell
order, and time labels. This makes their comparison a graph-threshold/training
control rather than a coordinate-alignment comparison. These compact runs are
not evidence that the released training code has been reproduced.

The primary package-recovery comparison uses `arista_spatial_full.yaml`, which
encodes the recovered six-stage CytoBridge spatial sequence:
`Pretrain(100) -> Refine(100) -> Init_interaction(50) -> Train_Score(2001) ->
Finetune(1000) -> Score_Refine(2001)`. It uses weighted OT coefficients
`10/0.05`, the recovered stage-specific global/reverse mass conventions,
forward-final-interval OT checkpoint rule for every neural-ODE stage (including
Finetune, whose best state is captured before the reverse update), last-state
score checkpoints, and the recovered plateau scheduler. Model
construction is seeded before initialization and
`last` checkpoints contain exactly the configured number of updates. Gradient
clipping at norm 10 is retained as a declared stabilization because the
unstabilized current ODE implementation produced non-finite finetuning states.
Downstream evaluation must report `score_stage=Score_Refine`; loading the
earlier `Train_Score` checkpoint is not a valid six-stage result.

The released checkpoint's `params.yml` records older top-level
`500/2001/1000` fields, but the original interaction-stage training driver and
execution log are not included with the released assets. Consequently this
matrix does not call the six-stage retrain a byte-for-byte historical replay.
It compares the published checkpoint against a separately labeled recovered
package implementation. Compact-profile results remain a labeled sensitivity
appendix/control.

The historical GNN source fixes the radial-basis centers and widths, and the
published ARISTA checkpoint values exactly equal those fixed initial values.
The recovered package therefore requires
`model.interaction_net.rbf_trainable: false`. Earlier runs that accidentally
optimized these parameters are retained only as `trainable-rbf` ablations and
are excluded from the formal reproduction table.

The recovered Finetune stage likewise uses `save_strategy: best` with
`checkpoint_metric: legacy_forward_last_ot`. The historical trainer saves the
lowest forward final-interval OT state before the reverse update; a run that
loads epoch 1,000's unconditional last state is a checkpoint-selection
ablation, not a formal reproduction condition.

The primary six-stage audit contains four separately labeled conditions:

1. `published_saved_model`: released checkpoint on the released 52-dimensional
   model-input table;
2. `recovered_six_stage_fixed_rbf_legacy_input`: current six-stage package code trained on
   the exact released `arista_1108_with_annotation.csv` state (`x1,x2` spatial;
   `x3..x52` latent; `0.05/0.45` thresholds);
3. `recovered_six_stage_fixed_rbf_current_preprocess_auto_thresholds`: current
   full-data preprocessing and the same training implementation with selected
   graph thresholds and the edge classifier fitted under that policy;
4. `recovered_six_stage_fixed_rbf_strict_frozen_auto_edge_published_thresholds`: an exact
   byte-for-byte copy of condition 3's aligned H5AD and edge-classifier weights,
   with only the fit-time spatial/edge decision thresholds changed to
   `0.05/0.45`.

Conditions 1→2 isolate checkpoint/training-implementation reproducibility as
far as the released model-input contract permits. Conditions 2→3 measure the
combined effect of current preprocessing/alignment, automatically selected
thresholds, and the current edge-classifier fit; they are not a single-factor
threshold comparison. The strict 3→4 comparison freezes both aligned H5AD and
edge weights and therefore isolates the fit-time cutoff/decision thresholds.
An independently preprocessed current-input `0.05/0.45` run is excluded from
the formal table because its changed edge-classifier weights would confound
that interpretation. The legacy CSV is imported through
`legacy_model_input_csv_to_adata`, which records that gene expression is absent
instead of presenting the table as a raw-data preprocessing result.

## Completed full-data audit

All four formal conditions were evaluated with the contract above. Mean values
across the five observed timepoints are:

| condition | joint W1 | joint W2 | PCA W1 | spatial W1 | spatial W2 | TMV | spatial NN dispersion | spatial support recall | spatial support precision | clump fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| published saved model | 3.15575 | 3.19708 | 3.06335 | 0.07377 | 0.08815 | 0.02321 | 0.9449 | 0.8785 | 0.8734 | 0.0110 |
| recovered six-stage, legacy input | 3.06626 | 3.10436 | 2.98822 | 0.06308 | 0.07515 | 0.01366 | 0.7981 | 0.8294 | 0.8970 | 0.0261 |
| recovered six-stage, current preprocess + automatic thresholds | 3.11900 | 3.15910 | 3.05329 | 0.06648 | 0.08182 | 0.00624 | 0.8966 | 0.8753 | 0.8882 | 0.0127 |
| same current input/edge weights, fit with `0.05/0.45` | 2.88505 | 2.92792 | 2.83097 | 0.06949 | 0.08095 | 0.00652 | 0.3128 | 0.4863 | 0.9193 | 0.0976 |

The strict threshold control has competitive transport metrics but repeated
local spatial collapse. The current-preprocess/automatic-threshold condition is
therefore the selected retrained model; the published checkpoint remains the
saved-model reference. The historical/current median nearest-neighbor distances
are `0.003160` and `0.003142`, with coordinate correlations `0.9977` and
`0.9900`. Thus the `0.05` versus `0.031543` spatial cutoffs are not explained by
a coordinate-scale change.

## Main Figure 5

| Panel | Manuscript content | Package-backed reproduction | Required output/status |
| --- | --- | --- | --- |
| 5a | Observed/generated stacked spatial map, reaEGC-focused lineage trajectories, and directed interaction edges | `notebooks/arista/arista_spatiotemporal_3d_api.ipynb` | **Full run complete:** 7,668 particles, piecewise warp, KNN `k=1`; attention interactions were recomputed through the API rather than loaded |
| 5b | Generated 3.5-DPI spatial slice | `notebooks/arista/arista_lineage_snapshot_api.ipynb` | **Full run complete:** generated t=0.5 model-time slice, corresponding to 3.5 DPI after dataset-time relabeling |
| 5c left | Spatial migration velocity streamlines | `notebooks/arista/arista_velocity_direction_correlation_api.ipynb` | **Full run complete:** `full_velocity_spatial.svg` at t=1 model time / 5 DPI |
| 5c right | Local cosine similarity between full spatial velocity and interaction-program velocity in the reaEGC injury-adjacent ROI | `notebooks/arista/arista_velocity_direction_correlation_api.ipynb` | **Full run complete:** ROI table and spatial panel; ROI definition and cell count are recorded in the manifest |
| 5d | Gene-expression velocity in PC1-PC2 space | `notebooks/arista/arista_velocity_direction_correlation_api.ipynb` | **Full run complete:** `full_velocity_pca.svg` at t=1 |
| 5e | Mean interaction magnitude versus mean growth for each time/cell-type group | `notebooks/arista/arista_growth_interaction_api.ipynb` | **Full run complete:** grouped table and bubble plot generated from model component calls |

The manuscript maps model times `0,1,2,3,4` to observed days post injury
`2,5,10,15,20`. Intermediate model times `0.5,1.5,2.5,3.5` therefore represent
the corresponding unobserved intervals (including 3.5 and 7.5 DPI in the
early interval descriptions). Every output must retain both the numeric model
time and, where displayed, the biological DPI label.

## Supplementary Figures S12-S17

| Figure | SI content | Reproduction path | API migration status |
| --- | --- | --- | --- |
| S12 | Observed and generated spatial maps on the dense 0.5-step grid | `arista_lineage_snapshot_api.ipynb` | **Full run complete** for the selected retrained model |
| S13 | Growth-rate maps on the dense time grid | `evaluate_growth_by_timepoint` + `plot_growth_timepoint_grid` on the shared interpolation slices | **Full run complete:** per-cell table and observed/generated dense-grid SVG are exported. Any injury marker must be supplied as explicit dataset metadata rather than inferred silently. |
| S14a | Lineage Sankey across observed and interpolated stages | `arista_lineage_snapshot_api.ipynb` | **Full run complete** for the selected retrained model |
| S14b | Cell-type composition across observed and interpolated stages | Classifier output + `summarize_label_composition` + `plot_celltype_composition` | **Full run complete:** long-form table and manuscript-style stacked-bar export are integrated into `arista_spatiotemporal_3d_api.ipynb` |
| S15 | Reconstructed temporal gene patterns and GO enrichment | `summarize_temporal_gene_patterns`, PCA inverse reconstruction, and temporal clustering | **Gene-pattern full run complete.** The repository does not yet contain a generic GO-enrichment backend/database contract, so the GO subpanel remains the only S15 item not internally recomputed. |
| S16 | Two ligand-receptor communication-pattern prototypes (22 and 46 pairs in the frozen SI) | Attention communication matrices + `project_communication_to_lr_timecourses` + temporal clustering | **Full parity runs complete.** Archived sidecars + first-symbol mapping reproduce the same 68-pair set. The frozen table is 22/46; published-checkpoint recomputation is 24/44 with 66/68 matching labels, showing two boundary pairs. Selected retrain is 11/57. |
| S17 | Small-multiple profiles for all 68 ligand-receptor pairs | Same LR time-course result as S16 with `plot_temporal_profile_small_multiples` | **Full run complete:** all 68 profiles export for both published and selected-retrained paper contracts. Across models, score Pearson/Spearman are 0.966/0.978 and median per-pair trajectory Pearson is 0.806; hard two-cluster ARI is -0.048. |

## Reviewer Figure 5c ROI analyses

The previous revision output is retained as an audit target, not silently
treated as a new package result. It reported 1,454/1,454 mapped ROI cells,
reaEGC enrichment of 259/259 cells at 5 DPI (5.57-fold; FDR
`1.5e-201`), WSN enrichment of 3.19-fold, MCG enrichment of 2.90-fold, and
marker genes including VIM, LGALS1, S100A10, CKB, and FTH1. The new
package-backed selected-model workflow exports 1,288 ROI cells with its
manifested automatic cutoff and bounds, rather than silently claiming the old
1,454-cell set. The spatial/PCA velocity and cosine panels are complete, but the
old enrichment, marker, and GO numbers should not be relabeled as selected-model
results without an explicit ROI-mapping sensitivity analysis.

## Provenance rules

- Every formal run writes a manifest with resolved absolute inputs, hashes,
  package import path, checkpoint stage, thresholds, seed, and evaluation
  settings.
- A cached classifier may be reused only when its feature, label, seed, and
  training metadata match the requested run.
- Legacy notebooks and `vendor/legacy_arista_stack` are migration references,
  not accepted runtime dependencies for the new panels.
- Published-checkpoint and current-checkpoint panels must be written to
  different output directories and clearly labeled.
- Benchmark numbers from held-out revision experiments are not interchangeable
  with the full-trajectory, no-holdout W1/W2/TMV comparison defined here.
