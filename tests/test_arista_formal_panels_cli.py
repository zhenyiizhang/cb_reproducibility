from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_arista_formal_panels_api.py"
)


@pytest.fixture()
def cli_module(monkeypatch):
    module_name = "_arista_formal_panels_cli_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_default_formal_contract_calls_shared_api(cli_module, monkeypatch, tmp_path):
    output_dir = tmp_path / "formal"
    snapshots = output_dir / "snapshots"
    manifest_path = _touch(output_dir / "run_manifest.json", "{}")
    spatiotemporal_html = _touch(output_dir / "spatiotemporal_3d.html")
    lineage_html = _touch(output_dir / "lineage_sankey.html")
    growth_csv = _touch(output_dir / "growth_dense_time_grid.csv")
    growth_figure = _touch(output_dir / "growth_dense_time_grid.svg")
    composition_csv = _touch(output_dir / "celltype_composition.csv")
    composition_figure = _touch(output_dir / "celltype_composition.svg")
    communications_pickle = _touch(output_dir / "all_time_communications.pkl")
    classifier_cache_dir = output_dir / "classifier_cache"
    _touch(snapshots / "time_0.5.svg")

    captured = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_run(config):
        captured["config"] = config
        _touch(classifier_cache_dir / "classifier.pt")
        return SimpleNamespace(
            manifest_path=manifest_path,
            snapshots_dir=snapshots,
            spatiotemporal_html=spatiotemporal_html,
            lineage_html=lineage_html,
            growth_csv=growth_csv,
            growth_figure=growth_figure,
            composition_csv=composition_csv,
            composition_figure=composition_figure,
            communications_pickle=communications_pickle,
            classifier_cache_dir=classifier_cache_dir,
        )

    fake_api = ModuleType("downstream_helpers.arista_api")
    fake_api.AristaSpatiotemporalConfig = FakeConfig
    fake_api.run_arista_spatiotemporal_api = fake_run
    fake_api.assert_package_only_runtime = lambda: None
    monkeypatch.setitem(sys.modules, "downstream_helpers.arista_api", fake_api)

    assert (
        cli_module.main(
            [
                "--aligned-h5ad",
                str(tmp_path / "aligned.h5ad"),
                "--model-dir",
                str(tmp_path / "training"),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    config = captured["config"]
    assert config.model_format == "current"
    assert config.time_points == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert config.interp_time_points == (0.5, 1.5, 2.5, 3.5)
    assert config.plot_3d_time_points == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert config.n_samples == 7668
    assert config.sde_dt == pytest.approx(0.05)
    assert config.split_sde_dt == pytest.approx(0.01)
    assert config.split_sigma == pytest.approx(0.03)
    assert config.random_seed == 42
    assert config.classifier_knn_neighbors == 1
    assert config.classifier_cache_path is None
    assert config.classifier_cache_dir == output_dir / "classifier_cache"
    assert config.spatial_warp_to_observed_piecewise is False
    assert config.spatial_warp_visualization_only is False
    assert config.skip_nonsplit_sde is False
    assert config.run_communication is True
    assert config.run_3d is True

    panel_index = json.loads(
        (output_dir / "formal_panel_index.json").read_text(encoding="utf-8")
    )
    assert panel_index["contract"]["spatial_warp_to_observed_piecewise"] is False
    assert panel_index["contract"]["classifier_cache_mode"] == "fresh_directory_fit"
    assert set(panel_index["panels"]) == {
        "Figure_5a",
        "Figure_5b",
        "Figure_S13",
        "Figure_S14a",
        "Figure_S14b",
    }
    assert str(snapshots / "time_0.5.svg") in panel_index["panels"]["Figure_5b"][
        "files"
    ]


def test_nonempty_cache_is_rejected_unless_reuse_is_explicit(cli_module, tmp_path):
    output_dir = tmp_path / "formal"
    _touch(output_dir / "classifier_cache" / "classifier.pt")

    with pytest.raises(FileExistsError, match="not empty"):
        cli_module.main(
            [
                "--aligned-h5ad",
                str(tmp_path / "aligned.h5ad"),
                "--model-dir",
                str(tmp_path / "training"),
                "--output-dir",
                str(output_dir),
            ]
        )


def test_time_grid_must_be_sorted_unique_and_cover_3d(cli_module):
    with pytest.raises(SystemExit):
        cli_module._parser().parse_args(
            [
                "--aligned-h5ad",
                "a.h5ad",
                "--model-dir",
                "model",
                "--output-dir",
                "out",
                "--observed-time-points",
                "0,2,1",
            ]
        )

    args = cli_module._parser().parse_args(
        [
            "--aligned-h5ad",
            "a.h5ad",
            "--model-dir",
            "model",
            "--output-dir",
            "out",
            "--plot-3d-time-points",
            "0,0.25",
        ]
    )
    with pytest.raises(ValueError, match="unknown"):
        cli_module._validate_args(args)
