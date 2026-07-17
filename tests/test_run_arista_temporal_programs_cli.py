from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_arista_temporal_programs_api.py"
)


def _load_cli(monkeypatch, calls):
    api = ModuleType("downstream_helpers.arista_api")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_run(config, **kwargs):
        calls.append((config, kwargs))
        return SimpleNamespace(manifest_path=Path("run_manifest.json"))

    api.AristaSpatiotemporalConfig = FakeConfig
    api.assert_package_only_runtime = lambda: None
    api.run_arista_temporal_programs_api = fake_run
    monkeypatch.setitem(sys.modules, "downstream_helpers.arista_api", api)

    name = "_run_arista_temporal_programs_cli_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, classifier_cache=None) -> argparse.Namespace:
    return argparse.Namespace(
        aligned_h5ad=str(tmp_path / "aligned.h5ad"),
        model_dir=str(tmp_path / "model"),
        lr_database=str(tmp_path / "lr.csv"),
        gene_set_gmt=str(tmp_path / "sets.gmt"),
        output_dir=str(tmp_path / "out"),
        device="cuda:7",
        seed=17,
        classifier_cache=classifier_cache,
    )


def test_clean_temporal_contract_uses_current_model_and_own_pca(
    monkeypatch, tmp_path
):
    calls = []
    cli = _load_cli(monkeypatch, calls)

    cli.run(_args(tmp_path))

    assert len(calls) == 1
    config, api_kwargs = calls[0]
    assert config.kwargs["model_format"] == "current"
    assert config.kwargs["time_points"] == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert config.kwargs["interp_time_points"] == (0.5, 1.5, 2.5, 3.5)
    assert config.kwargs["n_samples"] == 3072
    assert config.kwargs["skip_nonsplit_sde"] is True
    assert config.kwargs["split_sde_dt"] == 0.01
    assert config.kwargs["split_sigma"] == 0.03
    assert config.kwargs["spatial_warp_to_observed_piecewise"] is False
    assert config.kwargs["classifier_knn_neighbors"] == 10
    assert config.kwargs["run_communication"] is True
    assert config.kwargs["run_3d"] is False
    assert config.kwargs["random_seed"] == 17
    assert config.kwargs["classifier_cache_path"] is None
    assert config.kwargs["classifier_cache_dir"] is None

    assert api_kwargs["reference_h5ad"] == config.kwargs["aligned_h5ad"]
    assert api_kwargs["pca_components_csv"] is None
    assert api_kwargs["pca_center_csv"] is None
    assert api_kwargs["n_top_genes"] == 250
    assert api_kwargs["n_gene_cluster_genes"] == 2000
    assert api_kwargs["n_gene_clusters"] == 2
    assert api_kwargs["n_lr_clusters"] == 2
    assert api_kwargs["gene_set_background"] == "expression"
    assert api_kwargs["communication_random_seed"] == 17
    assert api_kwargs["preferred_species_tag"] is None
    assert api_kwargs["gene_profile_normalization"] == "zscore"
    assert api_kwargs["gene_profile_linkage_method"] == "average"
    assert api_kwargs["gene_profile_cluster_order"] == "raw"
    assert api_kwargs["lr_profile_linkage_method"] == "ward"
    assert api_kwargs["lr_profile_cluster_order"] == "dendrogram"
    assert api_kwargs["communication_max_cells_per_timepoint"] == 3072
    assert api_kwargs["communication_rng_warmup_max_cells_per_timepoint"] == 2500


def test_classifier_cache_accepts_file_or_directory(monkeypatch, tmp_path):
    calls = []
    cli = _load_cli(monkeypatch, calls)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = tmp_path / "classifier.pt"
    cache_file.touch()

    cli.run(_args(tmp_path, str(cache_dir)))
    cli.run(_args(tmp_path, str(cache_file)))

    directory_config = calls[0][0].kwargs
    file_config = calls[1][0].kwargs
    assert directory_config["classifier_cache_dir"] == cache_dir.resolve()
    assert directory_config["classifier_cache_path"] is None
    assert file_config["classifier_cache_path"] == cache_file.resolve()
    assert file_config["classifier_cache_dir"] is None
