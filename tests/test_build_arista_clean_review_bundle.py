from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_arista_clean_review_bundle.py"


def _load():
    spec = importlib.util.spec_from_file_location("_bundle_builder_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_builds_hashed_copy_only_bundle(monkeypatch, tmp_path):
    module = _load()
    run = tmp_path / "run"
    comparison = tmp_path / "comparison"
    clean = run / "clean.json"
    formal = run / "formal.json"
    display = run / "display.json"
    temporal = run / "temporal.json"
    split = comparison / "split.json"
    metrics = comparison / "metrics.csv"
    plan = [
        "Pretrain",
        "Refine",
        "Init_interaction",
        "Train_Score",
        "Finetune",
        "Score_Refine",
    ]
    _json(
        clean,
        {
            "preprocess_expression_source": "layers['counts']",
            "training_plan": [{"name": name} for name in plan],
            "score_stage": "Score_Refine",
            "rbf_trainable": False,
            "n_cells": 46189,
        },
    )
    _json(
        formal,
        {
            "config": {
                "spatial_warp_to_observed_piecewise": False,
                "classifier_knn_neighbors": 1,
                "skip_nonsplit_sde": False,
            },
            "trajectory_semantics": {"lineage_identity_source": "non_split_fixed_particles"},
        },
    )
    _json(
        display,
        {
            "spatial_warp_to_observed_piecewise": True,
            "spatial_warp_visualization_only": True,
            "video": {"enabled": True, "frames": 401},
        },
    )
    _json(temporal, {"config": {"classifier_knn_neighbors": 10}})
    _json(split, {"inputs": {f"condition_{index}": {} for index in range(5)}})
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "value"])
        writer.writeheader()
        for name in ("published", "legacy", "historical_double", "strict", "clean_counts"):
            writer.writerow({"model": name, "value": 1})

    specs = (
        module.ArtifactSpec("clean_run_manifest", "00_summary/clean.json", ("run",), ("clean.json",), "clean"),
        module.ArtifactSpec("formal_manifest", "06_formal_no_warp_sidecars/formal.json", ("run",), ("formal.json",), "formal"),
        module.ArtifactSpec("display_warp_manifest", "07_manifests/display.json", ("run",), ("display.json",), "display"),
        module.ArtifactSpec("temporal_manifest", "05_temporal_S15_S17/temporal.json", ("run",), ("temporal.json",), "temporal"),
        module.ArtifactSpec("five_condition_metrics_long", "01_five_condition_non_split/metrics.csv", ("comparison",), ("metrics.csv",), "metrics"),
        module.ArtifactSpec("five_condition_split_manifest", "02_five_condition_split/split.json", ("comparison",), ("split.json",), "split"),
    )
    monkeypatch.setattr(module, "SINGLE_ARTIFACTS", specs)
    monkeypatch.setattr(module, "COLLECTIONS", ())
    output = tmp_path / "bundle"
    result = module.build_bundle(
        run_root=run,
        output_dir=output,
        comparison_dir=comparison,
    )

    assert result["scientific_contract"]["formal_spatial_warp"] is False
    assert len(result["five_condition_non_split"]) == 5
    assert (output / "source_to_bundle_manifest.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert not any(path.is_symlink() for path in output.rglob("*"))
