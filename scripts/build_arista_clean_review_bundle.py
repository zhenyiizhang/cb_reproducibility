#!/usr/bin/env python3
"""Build a self-auditing ARISTA clean-counts manuscript review bundle.

The command deliberately copies (never symlinks) a compact set of figures,
tables, manifests, logs, and downstream sidecars from a completed run.  It
fails before creating the output directory when a required manuscript panel
or a required scientific contract is missing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence


GROUPS = (
    "00_summary",
    "01_five_condition_non_split",
    "02_five_condition_split",
    "03_main_figure5",
    "04_supplement_S12_S14",
    "05_temporal_S15_S17",
    "06_formal_no_warp_sidecars",
    "07_manifests",
    "08_logs",
)

CLEAN_SEMANTICS = (
    "Clean ARISTA counts-source run: layers['counts'] is normalized/log1p-transformed "
    "once; the fixed 2,000-HVG PCA contract is retained."
)
OLD_SEMANTICS = (
    "Five-condition audit. The pre-existing current-preprocess conditions are "
    "historical double-transform results and must not be relabelled as clean counts."
)
NON_SPLIT_SEMANTICS = (
    "Non-split combined-SDE evaluation; W1/W2/TMV use the saved particle-mass weights."
)
SPLIT_SEMANTICS = (
    "Split-SDE population display; growth is encoded by particle birth/death and point "
    "density, while marker area is fixed and does not encode particle weight."
)
FORMAL_SEMANTICS = (
    "Formal no-warp downstream state: non-split fixed particles define persistent "
    "lineages/ribbons, while split-SDE populations define generated slices and "
    "communication; classifier spatial refinement uses k=1 (disabled)."
)
DISPLAY_WARP_SEMANTICS = (
    "Display-only legacy piecewise spatial warp for the mosaic/video. Segment boundaries "
    "remain continuous and labels, communication, and subsequent dynamics use pre-warp "
    "state; this output is not an analysis coordinate system."
)
TEMPORAL_SEMANTICS = (
    "S15-S17 temporal paper-parity workflow; classifier spatial majority refinement uses "
    "k=10, separately from the formal k=1 lineage/communication workflow."
)


class BundleInputError(RuntimeError):
    """Raised when a strict review bundle cannot be built safely."""


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    dest: str
    roots: tuple[str, ...]
    patterns: tuple[str, ...]
    semantics: str


@dataclass(frozen=True)
class CollectionSpec:
    key: str
    dest_group: str
    roots: tuple[str, ...]
    patterns: tuple[str, ...]
    min_count: int
    semantics: str
    dest_mode: str = "condition"
    exclude_substrings: tuple[str, ...] = ()


SINGLE_ARTIFACTS = (
    ArtifactSpec(
        "clean_run_manifest",
        "00_summary/clean_counts_fixed2000_run_manifest.json",
        ("run",),
        ("downstream/run_manifest.json", "**/downstream/run_manifest.json"),
        CLEAN_SEMANTICS,
    ),
    ArtifactSpec(
        "clean_distribution_metrics",
        "00_summary/clean_counts_distribution_metrics.csv",
        ("run",),
        (
            "downstream/distribution_evaluation/distribution_metrics.csv",
            "**/distribution_evaluation/distribution_metrics.csv",
        ),
        f"{CLEAN_SEMANTICS} {NON_SPLIT_SEMANTICS}",
    ),
    ArtifactSpec(
        "five_condition_metrics_mean",
        "00_summary/five_condition_metrics_mean_by_space.csv",
        ("comparison", "run", "old_split"),
        ("comparison/model_metrics_mean_by_space.csv", "**/model_metrics_mean_by_space.csv"),
        f"{OLD_SEMANTICS} {NON_SPLIT_SEMANTICS}",
    ),
    ArtifactSpec(
        "five_condition_metrics_long",
        "01_five_condition_non_split/model_metrics_long.csv",
        ("comparison", "run", "old_split"),
        ("comparison/model_metrics_long.csv", "**/model_metrics_long.csv"),
        f"{OLD_SEMANTICS} {NON_SPLIT_SEMANTICS}",
    ),
    ArtifactSpec(
        "five_condition_metrics_deltas",
        "01_five_condition_non_split/model_metrics_paired_deltas.csv",
        ("comparison", "run", "old_split"),
        ("comparison/model_metrics_paired_deltas.csv", "**/model_metrics_paired_deltas.csv"),
        f"{OLD_SEMANTICS} {NON_SPLIT_SEMANTICS}",
    ),
    ArtifactSpec(
        "five_condition_metric_figure",
        "01_five_condition_non_split/model_metric_comparison.svg",
        ("comparison", "run", "old_split"),
        ("comparison/model_metric_comparison.svg", "**/model_metric_comparison.svg"),
        f"{OLD_SEMANTICS} {NON_SPLIT_SEMANTICS}",
    ),
    ArtifactSpec(
        "five_condition_local_structure_figure",
        "01_five_condition_non_split/model_local_structure_comparison.svg",
        ("comparison", "run", "old_split"),
        (
            "comparison/model_local_structure_comparison.svg",
            "**/model_local_structure_comparison.svg",
        ),
        f"{OLD_SEMANTICS} {NON_SPLIT_SEMANTICS}",
    ),
    ArtifactSpec(
        "five_condition_split_manifest",
        "02_five_condition_split/comparison_manifest.json",
        ("run", "comparison", "old_split"),
        (
            "split-comparison/comparison_manifest.json",
            "**/*split*comparison*/comparison_manifest.json",
            "comparison_manifest.json",
            "**/comparison_manifest.json",
        ),
        SPLIT_SEMANTICS,
    ),
    ArtifactSpec(
        "formal_manifest",
        "06_formal_no_warp_sidecars/formal_no_warp_manifest.json",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/manifest.json",
            "**/*formal*no*warp*/manifest.json",
            "spatiotemporal-formal-no-warp/run_manifest.json",
            "**/*formal*no*warp*/run_manifest.json",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "display_warp_manifest",
        "07_manifests/display_piecewise_warp_manifest.json",
        ("run",),
        (
            "spatiotemporal-display-piecewise-warp/manifest.json",
            "spatiotemporal-display-warp/manifest.json",
            "**/*display*warp*/manifest.json",
            "**/*piecewise*warp*/manifest.json",
        ),
        DISPLAY_WARP_SEMANTICS,
    ),
    ArtifactSpec(
        "figure5a_spatiotemporal",
        "03_main_figure5/Figure5a_spatiotemporal_3d_formal_no_warp.svg",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/spatiotemporal_3d.svg",
            "**/*formal*no*warp*/spatiotemporal_3d.svg",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "figure5b_generated_slice",
        "03_main_figure5/Figure5b_generated_t0p5_formal_no_warp.svg",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/snapshots/time_0.5.svg",
            "**/*formal*no*warp*/snapshots/time_0.5.svg",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "figure5c_spatial_velocity",
        "03_main_figure5/Figure5c_left_full_velocity_spatial.svg",
        ("run",),
        (
            "figure5cd/full_velocity_spatial.svg",
            "**/*figure5cd*/full_velocity_spatial.svg",
            "**/*direction*correlation*/full_velocity_spatial.svg",
        ),
        "Figure 5c left; full spatial velocity on the observed t=1 slice.",
    ),
    ArtifactSpec(
        "figure5c_roi_correlation",
        "03_main_figure5/Figure5c_right_full_vs_interaction_direction_roi.svg",
        ("run",),
        (
            "figure5cd/full_vs_interaction_direction_roi.svg",
            "**/*figure5cd*/full_vs_interaction_direction_roi.svg",
            "**/*direction*correlation*/full_vs_interaction_direction_roi.svg",
        ),
        "Figure 5c right; local full-versus-interaction spatial direction correlation.",
    ),
    ArtifactSpec(
        "figure5d_pca_velocity",
        "03_main_figure5/Figure5d_full_velocity_pca.svg",
        ("run",),
        (
            "figure5cd/full_velocity_pca.svg",
            "**/*figure5cd*/full_velocity_pca.svg",
            "**/*direction*correlation*/full_velocity_pca.svg",
        ),
        "Figure 5d; all-observed-cell gene velocity graph projected into global PC1-PC2.",
    ),
    ArtifactSpec(
        "figure5e_growth_interaction",
        "03_main_figure5/Figure5e_growth_interaction_celltype_bubble.svg",
        ("run",),
        (
            "figure5e/growth_interaction_celltype_bubble.svg",
            "**/*figure5e*/growth_interaction_celltype_bubble.svg",
            "**/*growth*interaction*/growth_interaction_celltype_bubble.svg",
        ),
        "Figure 5e; mean interaction magnitude versus mean growth by time and cell type.",
    ),
    ArtifactSpec(
        "s12_display_mosaic",
        "04_supplement_S12_S14/S12_dense_timepoint_mosaic_display_warp.svg",
        ("run",),
        (
            "spatiotemporal-display-piecewise-warp/snapshots/timepoint_mosaic.svg",
            "spatiotemporal-display-warp/snapshots/timepoint_mosaic.svg",
            "**/*display*warp*/snapshots/timepoint_mosaic.svg",
            "**/*piecewise*warp*/snapshots/timepoint_mosaic.svg",
        ),
        DISPLAY_WARP_SEMANTICS,
    ),
    ArtifactSpec(
        "s12_display_video",
        "04_supplement_S12_S14/S12_dense_split_population_video{source_suffix}",
        ("run",),
        (
            "spatiotemporal-display-piecewise-warp/spatiotemporal_split_population.mp4",
            "spatiotemporal-display-warp/spatiotemporal_split_population.mp4",
            "**/*display*warp*/spatiotemporal_split_population.mp4",
            "**/*piecewise*warp*/spatiotemporal_split_population.mp4",
            "**/*display*warp*/spatiotemporal_split_population.gif",
            "**/*piecewise*warp*/spatiotemporal_split_population.gif",
        ),
        DISPLAY_WARP_SEMANTICS,
    ),
    ArtifactSpec(
        "s13_growth_grid",
        "04_supplement_S12_S14/S13_growth_dense_time_grid.svg",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/growth_dense_time_grid.svg",
            "**/*formal*no*warp*/growth_dense_time_grid.svg",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s13_growth_table",
        "04_supplement_S12_S14/S13_growth_dense_time_grid.csv",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/growth_dense_time_grid.csv",
            "**/*formal*no*warp*/growth_dense_time_grid.csv",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s14a_lineage_sankey",
        "04_supplement_S12_S14/S14a_lineage_sankey_formal_no_warp.svg",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/lineage_sankey.svg",
            "**/*formal*no*warp*/lineage_sankey.svg",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s14b_celltype_composition_figure",
        "04_supplement_S12_S14/S14b_celltype_composition.svg",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/celltype_composition.svg",
            "**/*formal*no*warp*/celltype_composition.svg",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s14b_celltype_composition_table",
        "04_supplement_S12_S14/S14b_celltype_composition.csv",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/celltype_composition.csv",
            "**/*formal*no*warp*/celltype_composition.csv",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "temporal_manifest",
        "05_temporal_S15_S17/temporal_run_manifest.json",
        ("run",),
        (
            "temporal-s15-s17/run_manifest.json",
            "**/*temporal*S15*S17*/run_manifest.json",
            "**/*temporal*s15*s17*/run_manifest.json",
            "**/*temporal*program*/run_manifest.json",
            "**/*temporal*/run_manifest.json",
        ),
        TEMPORAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s15_gene_heatmap",
        "05_temporal_S15_S17/S15_gene_temporal_heatmap.svg",
        ("run",),
        ("temporal-s15-s17/figures/gene_temporal_heatmap.svg", "**/*temporal*/figures/gene_temporal_heatmap.svg"),
        TEMPORAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s15_gene_prototypes",
        "05_temporal_S15_S17/S15_gene_pattern_prototypes.svg",
        ("run",),
        ("temporal-s15-s17/figures/gene_pattern_prototypes.svg", "**/*temporal*/figures/gene_pattern_prototypes.svg"),
        TEMPORAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s15_gene_enrichment",
        "05_temporal_S15_S17/S15_gene_pattern_enrichment.csv",
        ("run",),
        ("temporal-s15-s17/tables/gene_pattern_enrichment.csv", "**/*temporal*/tables/gene_pattern_enrichment.csv"),
        TEMPORAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s16_lr_prototypes",
        "05_temporal_S15_S17/S16_lr_pattern_prototypes.svg",
        ("run",),
        ("temporal-s15-s17/figures/lr_pattern_prototypes.svg", "**/*temporal*/figures/lr_pattern_prototypes.svg"),
        TEMPORAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s16_lr_assignments",
        "05_temporal_S15_S17/S16_lr_pattern_assignments.csv",
        ("run",),
        ("temporal-s15-s17/tables/lr_pattern_assignments.csv", "**/*temporal*/tables/lr_pattern_assignments.csv"),
        TEMPORAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s17_lr_small_multiples",
        "05_temporal_S15_S17/S17_lr_pair_small_multiples.svg",
        ("run",),
        ("temporal-s15-s17/figures/lr_pair_small_multiples.svg", "**/*temporal*/figures/lr_pair_small_multiples.svg"),
        TEMPORAL_SEMANTICS,
    ),
    ArtifactSpec(
        "s17_lr_timecourse",
        "05_temporal_S15_S17/S17_lr_pair_timecourse.csv",
        ("run",),
        ("temporal-s15-s17/tables/lr_pair_timecourse.csv", "**/*temporal*/tables/lr_pair_timecourse.csv"),
        TEMPORAL_SEMANTICS,
    ),
    ArtifactSpec(
        "formal_communications",
        "06_formal_no_warp_sidecars/all_time_communications.pkl",
        ("run",),
        (
            "spatiotemporal-formal-no-warp/all_time_communications.pkl",
            "**/*formal*no*warp*/all_time_communications.pkl",
        ),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "formal_lineage_html",
        "06_formal_no_warp_sidecars/lineage_sankey.html",
        ("run",),
        ("spatiotemporal-formal-no-warp/lineage_sankey.html", "**/*formal*no*warp*/lineage_sankey.html"),
        FORMAL_SEMANTICS,
    ),
    ArtifactSpec(
        "formal_3d_html",
        "06_formal_no_warp_sidecars/spatiotemporal_3d.html",
        ("run",),
        ("spatiotemporal-formal-no-warp/spatiotemporal_3d.html", "**/*formal*no*warp*/spatiotemporal_3d.html"),
        FORMAL_SEMANTICS,
    ),
)


COLLECTIONS = (
    CollectionSpec(
        "five_non_split_spatial_panels",
        "01_five_condition_non_split",
        ("run", "comparison", "old_split"),
        ("**/generated_vs_observed_spatial.svg",),
        5,
        f"{OLD_SEMANTICS} {NON_SPLIT_SEMANTICS}",
        exclude_substrings=("_split",),
    ),
    CollectionSpec(
        "five_non_split_pca_panels",
        "01_five_condition_non_split",
        ("run", "comparison", "old_split"),
        ("**/generated_vs_observed_pca.svg",),
        5,
        f"{OLD_SEMANTICS} {NON_SPLIT_SEMANTICS}",
        exclude_substrings=("_split",),
    ),
    CollectionSpec(
        "five_split_spatial_panels",
        "02_five_condition_split",
        ("run", "comparison", "old_split"),
        ("**/generated_vs_observed_spatial_split.svg",),
        5,
        SPLIT_SEMANTICS,
    ),
    CollectionSpec(
        "five_split_pca_panels",
        "02_five_condition_split",
        ("run", "comparison", "old_split"),
        ("**/generated_vs_observed_pca_split.svg",),
        5,
        SPLIT_SEMANTICS,
    ),
    CollectionSpec(
        "s15_enrichment_figures",
        "05_temporal_S15_S17",
        ("run",),
        ("**/*temporal*/figures/gene_pattern_*_enrichment_*.svg",),
        2,
        TEMPORAL_SEMANTICS,
        dest_mode="basename",
    ),
    CollectionSpec(
        "formal_attention_arrays",
        "06_formal_no_warp_sidecars",
        ("run",),
        ("**/*formal*no*warp*/attention/*.npy",),
        2,
        FORMAL_SEMANTICS,
        dest_mode="basename",
    ),
    CollectionSpec(
        "formal_classifier_cache",
        "06_formal_no_warp_sidecars",
        ("run",),
        ("**/*formal*no*warp*/classifier_cache/*.pt",),
        1,
        f"{FORMAL_SEMANTICS} Classifier metadata are embedded in the cache and formal manifest.",
        dest_mode="basename",
    ),
    CollectionSpec(
        "provenance_manifests",
        "07_manifests",
        ("run", "comparison", "old_split"),
        ("**/*manifest*.json", "**/*.meta.json", "**/params.yml"),
        4,
        "Copied provenance/configuration metadata; paths and hashes remain auditable in the bundle index.",
        dest_mode="root_relative",
    ),
    CollectionSpec(
        "run_logs",
        "08_logs",
        ("run",),
        ("**/*.log", "**/logs/*.txt"),
        1,
        "Execution logs from the clean-counts preprocessing, training, evaluation, and panel run.",
        dest_mode="root_relative",
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--old-split-bundle-dir", type=Path, default=None)
    parser.add_argument("--comparison-dir", type=Path, default=None)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_root(path: Path | None, label: str, *, required: bool) -> Path | None:
    if path is None:
        if required:
            raise BundleInputError(f"Missing required {label}.")
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise BundleInputError(f"{label} is not a directory: {resolved}")
    return resolved


def _matches(root: Path, pattern: str) -> list[Path]:
    return sorted(
        {path.resolve() for path in root.glob(pattern) if path.is_file()},
        key=lambda value: str(value),
    )


def _static_pattern_variants(pattern: str) -> tuple[str, ...]:
    """Allow a formal static panel to be supplied as SVG, PDF, or PNG."""
    if pattern.lower().endswith(".svg"):
        stem = pattern[:-4]
        return (f"{stem}.svg", f"{stem}.pdf", f"{stem}.png")
    return (pattern,)


def _resolve_single(
    spec: ArtifactSpec,
    roots: Mapping[str, Path | None],
) -> tuple[str, Path, Path] | str:
    searched: list[str] = []
    for root_name in spec.roots:
        root = roots.get(root_name)
        if root is None:
            continue
        for base_pattern in spec.patterns:
            for pattern in _static_pattern_variants(base_pattern):
                searched.append(str(root / pattern))
                matches = _matches(root, pattern)
                if len(matches) == 1:
                    return root_name, matches[0], root
                if len(matches) > 1:
                    rendered = "\n      ".join(str(path) for path in matches)
                    return (
                        f"{spec.key}: candidate pattern is ambiguous: {root / pattern}\n"
                        f"      {rendered}"
                    )
    rendered_search = "\n      ".join(searched) if searched else "<no configured source root>"
    return f"{spec.key}: no required file found; searched:\n      {rendered_search}"


def _resolve_collection(
    spec: CollectionSpec,
    roots: Mapping[str, Path | None],
) -> tuple[list[tuple[str, Path, Path]], str | None]:
    found: list[tuple[str, Path, Path]] = []
    seen: set[Path] = set()
    searched: list[str] = []
    for root_name in spec.roots:
        root = roots.get(root_name)
        if root is None:
            continue
        for base_pattern in spec.patterns:
            # Prefer one static representation per logical panel. This prevents
            # SVG+PNG exports of one condition from satisfying a five-condition check.
            selected_by_stem: dict[Path, Path] = {}
            for pattern in _static_pattern_variants(base_pattern):
                searched.append(str(root / pattern))
                for path in _matches(root, pattern):
                    lower_path = str(path).lower()
                    if any(token.lower() in lower_path for token in spec.exclude_substrings):
                        continue
                    logical_stem = path.with_suffix("")
                    selected_by_stem.setdefault(logical_stem, path)
            for path in selected_by_stem.values():
                if path not in seen:
                    found.append((root_name, path, root))
                    seen.add(path)
    if len(found) < spec.min_count:
        rendered_search = "\n      ".join(searched) if searched else "<no configured source root>"
        return found, (
            f"{spec.key}: found {len(found)} file(s), require at least {spec.min_count}; "
            f"searched:\n      {rendered_search}"
        )
    return found, None


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleInputError(f"Cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleInputError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_run_manifest(path: Path) -> list[str]:
    data = _load_json(path, "clean run manifest")
    errors: list[str] = []
    expression_source = str(data.get("preprocess_expression_source", ""))
    if "counts" not in expression_source.lower():
        errors.append(
            "clean_run_manifest: preprocess_expression_source must be layers['counts']; "
            f"received {expression_source!r}"
        )
    expected_plan = [
        "Pretrain",
        "Refine",
        "Init_interaction",
        "Train_Score",
        "Finetune",
        "Score_Refine",
    ]
    actual_plan = [str(stage.get("name")) for stage in data.get("training_plan", [])]
    if actual_plan != expected_plan:
        errors.append(
            f"clean_run_manifest: expected six-stage plan {expected_plan}, received {actual_plan}"
        )
    if data.get("score_stage") != "Score_Refine":
        errors.append(
            "clean_run_manifest: downstream score_stage must be 'Score_Refine', "
            f"received {data.get('score_stage')!r}"
        )
    if data.get("rbf_trainable") is not False:
        errors.append(
            "clean_run_manifest: formal recovered training requires rbf_trainable=false"
        )
    if int(data.get("n_cells", 0) or 0) < 40000:
        errors.append(
            "clean_run_manifest: fewer than 40,000 cells indicates a smoke/subsampled run"
        )
    return errors


def _manifest_value(data: Mapping[str, object], key: str) -> object:
    if key in data:
        return data[key]
    config = data.get("config")
    return config.get(key) if isinstance(config, Mapping) else None


def _validate_formal_manifest(path: Path) -> list[str]:
    data = _load_json(path, "formal no-warp manifest")
    errors: list[str] = []
    if bool(_manifest_value(data, "spatial_warp_to_observed_piecewise")):
        errors.append("formal_manifest: spatial_warp_to_observed_piecewise must be false")
    if int(_manifest_value(data, "classifier_knn_neighbors") or 0) != 1:
        errors.append("formal_manifest: classifier_knn_neighbors must be 1")
    if bool(_manifest_value(data, "skip_nonsplit_sde")):
        errors.append("formal_manifest: non-split SDE must be retained for lineage identity")
    semantics = data.get("trajectory_semantics", {})
    if isinstance(semantics, Mapping):
        lineage = semantics.get("lineage_identity_source")
        if lineage not in (None, "non_split_fixed_particles"):
            errors.append(
                "formal_manifest: lineage_identity_source must be non_split_fixed_particles"
            )
    return errors


def _validate_display_manifest(path: Path) -> list[str]:
    data = _load_json(path, "display warp manifest")
    errors: list[str] = []
    if not bool(_manifest_value(data, "spatial_warp_to_observed_piecewise")):
        errors.append("display_warp_manifest: piecewise warp must be enabled")
    if not bool(_manifest_value(data, "spatial_warp_visualization_only")):
        errors.append("display_warp_manifest: warp must be visualization-only")
    video = data.get("video", {})
    if isinstance(video, Mapping):
        frames = int(video.get("frames", 0) or 0)
        if not bool(video.get("enabled")) or frames < 9:
            errors.append(
                "display_warp_manifest: a dense video with at least 9 frames is required"
            )
    else:
        errors.append("display_warp_manifest: video provenance is missing")
    return errors


def _validate_temporal_manifest(path: Path) -> list[str]:
    data = _load_json(path, "temporal manifest")
    neighbors = _manifest_value(data, "classifier_knn_neighbors")
    if neighbors is None:
        classifier = data.get("classifier")
        if isinstance(classifier, Mapping):
            neighbors = classifier.get("knn_neighbors")
    if int(neighbors or 0) != 10:
        return ["temporal_manifest: paper-parity classifier_knn_neighbors must be 10"]
    return []


def _csv_models(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "model" not in reader.fieldnames:
            raise BundleInputError(f"Five-condition metrics table lacks a 'model' column: {path}")
        return sorted({str(row["model"]).strip() for row in reader if str(row["model"]).strip()})


def _validate_five_conditions(metrics_path: Path, split_manifest_path: Path) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    models = _csv_models(metrics_path)
    if len(models) != 5:
        errors.append(
            f"five_condition_metrics_long: expected exactly 5 models, found {len(models)}: {models}"
        )
    if not any("clean" in name.lower() and "count" in name.lower() for name in models):
        errors.append(
            "five_condition_metrics_long: one model name must explicitly identify the clean-counts condition"
        )
    split = _load_json(split_manifest_path, "five-condition split manifest")
    raw_inputs = split.get("inputs", {})
    split_models = sorted(map(str, raw_inputs)) if isinstance(raw_inputs, Mapping) else []
    if len(split_models) != 5:
        errors.append(
            f"five_condition_split_manifest: expected exactly 5 inputs, found {len(split_models)}: {split_models}"
        )
    return models, split_models, errors


def _render_dest(template: str, source: Path) -> Path:
    rendered = Path(template.format(source_suffix=source.suffix.lower()))
    if rendered.suffix.lower() in {".svg", ".pdf", ".png"} and source.suffix.lower() in {
        ".svg",
        ".pdf",
        ".png",
    }:
        rendered = rendered.with_suffix(source.suffix.lower())
    return rendered


def _condition_name(root_name: str, path: Path) -> str:
    if root_name == "run" and path.parent.name == "distribution_evaluation":
        return "clean_counts_fixed2000"
    return path.parent.name or root_name


def _collection_dest(
    spec: CollectionSpec,
    root_name: str,
    source: Path,
    root: Path,
) -> Path:
    if spec.dest_mode == "basename":
        return Path(spec.dest_group) / source.name
    if spec.dest_mode == "root_relative":
        return Path(spec.dest_group) / root_name / source.relative_to(root)
    if spec.dest_mode == "condition":
        return Path(spec.dest_group) / _condition_name(root_name, source) / source.name
    raise ValueError(f"Unknown collection destination mode: {spec.dest_mode}")


def _deduplicate_destination(dest: Path, source: Path, used: Mapping[Path, Path]) -> Path:
    if dest not in used or used[dest] == source:
        return dest
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
    return dest.with_name(f"{dest.stem}__{digest}{dest.suffix}")


def _copy_verified(source: Path, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink():
        raise RuntimeError(f"Bundle destination unexpectedly became a symlink: {destination}")
    source_hash = _sha256(source)
    destination_hash = _sha256(destination)
    if source_hash != destination_hash:
        raise IOError(f"Copy checksum mismatch: {source} -> {destination}")
    return source_hash, int(destination.stat().st_size)


def _readme(
    run_root: Path,
    models: Sequence[str],
    split_models: Sequence[str],
    n_files: int,
) -> str:
    return f"""# ARISTA clean-counts full-run review bundle

This directory is a copied, checksum-verifiable review bundle for the full ARISTA
preprocessing, six-stage training, evaluation, downstream analysis, and manuscript
panels. It contains {n_files} copied source artifacts. The canonical run remains at
`{run_root}`; this bundle does not contain symlinks and does not modify that run.

## Scientific contracts

- **Clean preprocessing:** `layers['counts']` is the expression source, transformed
  exactly once. The recovered analysis retains the fixed 2,000-HVG PCA contract; it
  is therefore a **counts-source, fixed-2000 clean rerun**, not a full-gene raw rerun.
- **Historical comparison:** the older current-preprocess/automatic-threshold and
  paired frozen-edge conditions were produced before the counts-source correction
  and are retained as historical **double-transform** conditions. They are comparison
  controls, not clean-counts outputs.
- **Formal downstream:** lineage ribbons use non-split fixed-particle identities;
  generated populations and communication use split SDE. Spatial piecewise warp is
  disabled, and classifier spatial refinement is `k=1` (effectively disabled).
- **Display-only warp:** the S12 mosaic/video and Figure 5b slice may use the legacy
  continuous piecewise spatial warp. Labels, communication, and subsequent dynamics
  continue from pre-warp state; the warped coordinates are presentation-only.
- **Temporal S15-S17:** paper-parity temporal classification uses spatial KNN `k=10`.
  This is intentionally different from the formal lineage/communication `k=1` run.
- **Weights:** non-split W1/W2/TMV evaluation uses particle-mass weights. Split-SDE
  plots use fixed marker area; population mass is represented by point count/density.

## Five conditions

Non-split metric-table models:

{chr(10).join(f'- `{name}`' for name in models)}

Split-panel models:

{chr(10).join(f'- `{name}`' for name in split_models)}

## Layout

- `00_summary`: clean-run and five-condition summary tables/provenance.
- `01_five_condition_non_split`: weighted non-split W1/W2/TMV, local structure,
  and generated-versus-observed PCA/spatial panels.
- `02_five_condition_split`: matched fixed-marker split-SDE PCA/spatial panels.
- `03_main_figure5`: Figure 5a-e panel exports.
- `04_supplement_S12_S14`: dense maps/video, growth, Sankey, and composition.
- `05_temporal_S15_S17`: temporal gene, enrichment, and LR pattern outputs.
- `06_formal_no_warp_sidecars`: formal trajectories, communication, classifier,
  attention, and interactive figures.
- `07_manifests`: copied run/configuration manifests.
- `08_logs`: execution logs.

`source_to_bundle_manifest.json` records every copied source, destination, SHA-256,
and interpretation. `SHA256SUMS` verifies every file in this bundle except itself.
"""


def build_bundle(
    *,
    run_root: Path,
    output_dir: Path,
    old_split_bundle_dir: Path | None = None,
    comparison_dir: Path | None = None,
) -> dict:
    """Resolve, validate, and copy one complete strict review bundle."""
    run = _require_root(run_root, "--run-root", required=True)
    assert run is not None
    roots: dict[str, Path | None] = {
        "run": run,
        "comparison": _require_root(comparison_dir, "--comparison-dir", required=False),
        "old_split": _require_root(
            old_split_bundle_dir, "--old-split-bundle-dir", required=False
        ),
    }
    output = output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise BundleInputError(
            f"--output-dir must be absent or empty so an old bundle is never mixed in: {output}"
        )

    resolved_singles: dict[str, tuple[ArtifactSpec, str, Path, Path]] = {}
    errors: list[str] = []
    for spec in SINGLE_ARTIFACTS:
        result = _resolve_single(spec, roots)
        if isinstance(result, str):
            errors.append(result)
        else:
            root_name, source, root = result
            resolved_singles[spec.key] = (spec, root_name, source, root)

    resolved_collections: dict[str, tuple[CollectionSpec, list[tuple[str, Path, Path]]]] = {}
    for spec in COLLECTIONS:
        values, error = _resolve_collection(spec, roots)
        if error:
            errors.append(error)
        resolved_collections[spec.key] = (spec, values)

    if errors:
        raise BundleInputError(
            "Strict ARISTA review bundle inputs are incomplete:\n- "
            + "\n- ".join(errors)
        )

    errors.extend(_validate_run_manifest(resolved_singles["clean_run_manifest"][2]))
    errors.extend(_validate_formal_manifest(resolved_singles["formal_manifest"][2]))
    errors.extend(_validate_display_manifest(resolved_singles["display_warp_manifest"][2]))
    errors.extend(_validate_temporal_manifest(resolved_singles["temporal_manifest"][2]))
    models, split_models, condition_errors = _validate_five_conditions(
        resolved_singles["five_condition_metrics_long"][2],
        resolved_singles["five_condition_split_manifest"][2],
    )
    errors.extend(condition_errors)
    if errors:
        raise BundleInputError(
            "Strict ARISTA review bundle semantic validation failed:\n- "
            + "\n- ".join(errors)
        )

    output.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        (output / group).mkdir(parents=True, exist_ok=True)

    copy_plan: list[tuple[str, Path, Path, str]] = []
    used_destinations: dict[Path, Path] = {}
    for spec, _root_name, source, _root in resolved_singles.values():
        dest = _render_dest(spec.dest, source)
        dest = _deduplicate_destination(dest, source, used_destinations)
        used_destinations[dest] = source
        copy_plan.append((spec.key, source, dest, spec.semantics))
    for spec, sources in resolved_collections.values():
        for root_name, source, source_root in sources:
            dest = _collection_dest(spec, root_name, source, source_root)
            dest = _deduplicate_destination(dest, source, used_destinations)
            used_destinations[dest] = source
            copy_plan.append((spec.key, source, dest, spec.semantics))

    records: list[dict[str, object]] = []
    for logical_id, source, relative_dest, semantics in sorted(
        copy_plan, key=lambda item: str(item[2])
    ):
        digest, size = _copy_verified(source, output / relative_dest)
        records.append(
            {
                "logical_id": logical_id,
                "source": str(source.resolve()),
                "dest": relative_dest.as_posix(),
                "sha256": digest,
                "size_bytes": size,
                "semantics": semantics,
            }
        )

    index = {
        "workflow": "build_arista_clean_review_bundle",
        "run_root": str(run),
        "output_dir": str(output),
        "scientific_contract": {
            "preprocessing": "counts-source, fixed-2000-HVG PCA, single transform",
            "historical_old_current_preprocess": "double transform; comparison only",
            "formal_spatial_warp": False,
            "display_piecewise_warp": "visualization only",
            "formal_classifier_knn_neighbors": 1,
            "temporal_classifier_knn_neighbors": 10,
            "non_split_weights": "used by W1/W2/TMV",
            "split_marker_area": "fixed; weights not encoded by size",
        },
        "five_condition_non_split": models,
        "five_condition_split": split_models,
        "n_copied_files": len(records),
        "files": records,
    }
    index_path = output / "source_to_bundle_manifest.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    readme_path = output / "README.md"
    readme_path.write_text(
        _readme(run, models, split_models, len(records)), encoding="utf-8"
    )

    checksum_paths = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    sums = "".join(
        f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n"
        for path in checksum_paths
    )
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8")

    symlinks = [path for path in output.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"Bundle unexpectedly contains symlinks: {symlinks}")
    return index


def main() -> int:
    args = _parser().parse_args()
    index = build_bundle(
        run_root=args.run_root,
        output_dir=args.output_dir,
        old_split_bundle_dir=args.old_split_bundle_dir,
        comparison_dir=args.comparison_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": index["output_dir"],
                "n_copied_files": index["n_copied_files"],
                "manifest": str(Path(index["output_dir"]) / "source_to_bundle_manifest.json"),
                "checksums": str(Path(index["output_dir"]) / "SHA256SUMS"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
