from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_arista_clean_review_bundle import BundleInputError, GROUPS, build_bundle


def _write(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, value: dict) -> Path:
    return _write(path, json.dumps(value, indent=2, sort_keys=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    comparison = tmp_path / "comparison_bundle"
    old_split = tmp_path / "old_split_bundle"

    stage_names = [
        "Pretrain",
        "Refine",
        "Init_interaction",
        "Train_Score",
        "Finetune",
        "Score_Refine",
    ]
    _json(
        run / "downstream" / "run_manifest.json",
        {
            "workflow": "arista_end_to_end",
            "preprocess_expression_source": "layers['counts']",
            "training_plan": [{"name": name} for name in stage_names],
            "score_stage": "Score_Refine",
            "rbf_trainable": False,
            "n_cells": 46189,
        },
    )
    distribution = run / "downstream" / "distribution_evaluation"
    _write(distribution / "distribution_metrics.csv", "time,space,w1,w2,tmv\n0,joint,1,1,0\n")
    _write(distribution / "generated_vs_observed_spatial.svg")
    _write(distribution / "generated_vs_observed_pca.svg")

    models = [
        "published_saved_model",
        "recovered_six_stage_fixed_rbf_legacy_input",
        "historical_current_preprocess_auto_thresholds",
        "historical_strict_frozen_edge_published_thresholds",
        "clean_counts_fixed2000_six_stage_auto_thresholds",
    ]
    comparison_api = comparison / "comparison"
    rows = "model,time,space,w1,w2,tmv\n" + "".join(
        f"{model},0,joint,1,1,0\n" for model in models
    )
    _write(comparison_api / "model_metrics_long.csv", rows)
    _write(comparison_api / "model_metrics_mean_by_space.csv", "model,space,w1\n")
    _write(comparison_api / "model_metrics_paired_deltas.csv", "candidate,baseline,w1_delta\n")
    _write(comparison_api / "model_metric_comparison.svg")
    _write(comparison_api / "model_local_structure_comparison.svg")
    for index in range(4):
        condition = comparison / f"historical_condition_{index + 1}"
        _write(condition / "generated_vs_observed_spatial.svg")
        _write(condition / "generated_vs_observed_pca.svg")

    clean_split = run / "split-panels" / "clean_counts_fixed2000"
    _write(clean_split / "generated_vs_observed_spatial_split.svg")
    _write(clean_split / "generated_vs_observed_pca_split.svg")
    for index in range(4):
        condition = old_split / f"historical_condition_{index + 1}"
        _write(condition / "generated_vs_observed_spatial_split.svg")
        _write(condition / "generated_vs_observed_pca_split.svg")
    _json(
        run / "split-comparison" / "comparison_manifest.json",
        {"workflow": "render_arista_split_distribution_comparison", "inputs": {name: {} for name in models}},
    )

    formal = run / "spatiotemporal-formal-no-warp"
    _json(
        formal / "manifest.json",
        {
            "spatial_warp_to_observed_piecewise": False,
            "classifier_knn_neighbors": 1,
            "skip_nonsplit_sde": False,
            "trajectory_semantics": {"lineage_identity_source": "non_split_fixed_particles"},
        },
    )
    for relative in (
        "spatiotemporal_3d.svg",
        "snapshots/time_0.5.svg",
        "growth_dense_time_grid.svg",
        "growth_dense_time_grid.csv",
        "lineage_sankey.svg",
        "celltype_composition.svg",
        "celltype_composition.csv",
        "all_time_communications.pkl",
        "split_population_trajectory.npy",
        "split_population_labels.npy",
        "lineage_sankey.html",
        "spatiotemporal_3d.html",
        "attention/attn_mean_t0.npy",
        "attention/edge_index_t0.npy",
        "classifier_cache/classifier_resmlp_fixture.pt",
    ):
        _write(formal / relative)

    display = run / "spatiotemporal-display-piecewise-warp"
    _json(
        display / "manifest.json",
        {
            "spatial_warp_to_observed_piecewise": True,
            "spatial_warp_visualization_only": True,
            "video": {"enabled": True, "frames": 17},
        },
    )
    _write(display / "snapshots" / "time_0.5.svg")
    _write(display / "snapshots" / "timepoint_mosaic.svg")
    _write(display / "spatiotemporal_split_population.mp4")

    figure5cd = run / "figure5cd"
    _write(figure5cd / "full_velocity_spatial.svg")
    _write(figure5cd / "full_vs_interaction_direction_roi.svg")
    _write(figure5cd / "full_velocity_pca.svg")
    _json(figure5cd / "run_manifest.json", {"workflow": "arista_direction_correlation_api"})
    figure5e = run / "figure5e"
    _write(figure5e / "growth_interaction_celltype_bubble.svg")
    _json(figure5e / "run_manifest.json", {"workflow": "arista_growth_interaction_api"})

    temporal = run / "temporal-s15-s17"
    _json(
        temporal / "run_manifest.json",
        {
            "workflow": "arista_temporal_programs_api",
            "config": {"classifier_knn_neighbors": 10},
        },
    )
    for relative in (
        "figures/gene_temporal_heatmap.svg",
        "figures/gene_pattern_prototypes.svg",
        "figures/gene_pattern_0_enrichment_dot.svg",
        "figures/gene_pattern_1_enrichment_dot.svg",
        "figures/lr_pattern_prototypes.svg",
        "figures/lr_pair_small_multiples.svg",
        "tables/gene_pattern_enrichment.csv",
        "tables/lr_pattern_assignments.csv",
        "tables/lr_pair_timecourse.csv",
    ):
        _write(temporal / relative)

    _write(run / "logs" / "full_pipeline.log", "preprocess train downstream complete\n")
    return run, comparison, old_split


def test_build_bundle_copies_and_hashes_all_required_outputs(tmp_path: Path) -> None:
    run, comparison, old_split = _complete_sources(tmp_path)
    output = tmp_path / "review_bundle"

    index = build_bundle(
        run_root=run,
        output_dir=output,
        comparison_dir=comparison,
        old_split_bundle_dir=old_split,
    )

    assert index["n_copied_files"] == len(index["files"])
    assert index["n_copied_files"] > 40
    assert all((output / group).is_dir() for group in GROUPS)
    assert not any(path.is_symlink() for path in output.rglob("*"))
    for record in index["files"]:
        source = Path(record["source"])
        destination = output / record["dest"]
        assert source.is_absolute()
        assert destination.is_file()
        assert record["sha256"] == _sha256(source) == _sha256(destination)
        assert record["semantics"]

    saved_index = json.loads(
        (output / "source_to_bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_index["scientific_contract"]["formal_spatial_warp"] is False
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "double-transform" in readme
    assert "`k=10`" in readme
    assert "presentation-only" in readme
    checksums = (output / "SHA256SUMS").read_text(encoding="utf-8")
    assert "README.md" in checksums
    assert "source_to_bundle_manifest.json" in checksums
    assert "SHA256SUMS" not in checksums


def test_static_panel_falls_back_to_png_without_renaming_as_svg(tmp_path: Path) -> None:
    run, comparison, old_split = _complete_sources(tmp_path)
    svg = run / "figure5cd" / "full_velocity_pca.svg"
    png = svg.with_suffix(".png")
    svg.rename(png)
    output = tmp_path / "review_bundle"

    build_bundle(
        run_root=run,
        output_dir=output,
        comparison_dir=comparison,
        old_split_bundle_dir=old_split,
    )

    assert (output / "03_main_figure5" / "Figure5d_full_velocity_pca.png").is_file()
    assert not (output / "03_main_figure5" / "Figure5d_full_velocity_pca.svg").exists()


def test_missing_panel_reports_logical_name_and_candidates_before_writing(tmp_path: Path) -> None:
    run, comparison, old_split = _complete_sources(tmp_path)
    (run / "figure5e" / "growth_interaction_celltype_bubble.svg").unlink()
    output = tmp_path / "review_bundle"

    with pytest.raises(BundleInputError) as excinfo:
        build_bundle(
            run_root=run,
            output_dir=output,
            comparison_dir=comparison,
            old_split_bundle_dir=old_split,
        )

    message = str(excinfo.value)
    assert "figure5e_growth_interaction" in message
    assert "growth_interaction_celltype_bubble" in message
    assert not output.exists()
