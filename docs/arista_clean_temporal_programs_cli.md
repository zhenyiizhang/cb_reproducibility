# Clean ARISTA temporal-program CLI (S15-S17)

`scripts/run_arista_temporal_programs_api.py` regenerates the prospective
temporal gene and ligand-receptor panels through
`downstream_helpers.arista_api.run_arista_temporal_programs_api`.

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_arista_temporal_programs_api.py \
  --aligned-h5ad /path/to/preprocess/arista_aligned.h5ad \
  --model-dir /path/to/training \
  --lr-database /path/to/CellChatDB.ligrec.human.csv \
  --gene-set-gmt /path/to/GO_Biological_Process_2023.gmt \
  --output-dir /path/to/review/temporal_programs_s15_s17 \
  --device cuda \
  --seed 42 \
  --classifier-cache /path/to/compatible/classifier_cache
```

The classifier-cache option accepts either one compatible checkpoint file or a
cache directory. It may be omitted to train a classifier and cache it inside
the output directory.

This command deliberately defines one clean prospective contract: current
model format; observed times 0-4 and half-step interpolations; 3,072 particles;
split SDE only with `dt=0.01` and `sigma=0.03`; no spatial warp; classifier
KNN=10; communication enabled and 3D disabled; 250 displayed genes, all 2,000
reconstructed genes assigned to two temporal clusters, and two LR clusters;
offline GMT enrichment against the reconstructed expression background.
Gene profiles use z-score/average-linkage with raw cluster ordering; LR
profiles use Ward linkage and dendrogram ordering. Communication is capped at
3,072 cells per time point after the documented 2,500-cell RNG warm-up, and
the paper-parity LR symbol mapping does not silently prefer a species-tagged
duplicate.

The aligned H5AD is used as both the model input and gene-expression reference.
It must contain its own PCA loadings in `varm['PCs']`. The CLI exposes no PCA
components/center sidecar arguments and passes both legacy sidecars as `None`,
so a missing package-native inverse-PCA contract fails instead of silently
borrowing the published checkpoint's PCA basis.
