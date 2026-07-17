from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_arista_growth_interaction_api.py"


def _load(monkeypatch, calls):
    api = ModuleType("downstream_helpers.arista_api")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_run(config, **kwargs):
        calls.append((config, kwargs))
        return SimpleNamespace(manifest_path=Path("run_manifest.json"))

    api.AristaSpatiotemporalConfig = FakeConfig
    api.assert_package_only_runtime = lambda: None
    api.run_arista_growth_interaction_api = fake_run
    monkeypatch.setitem(sys.modules, "downstream_helpers.arista_api", api)
    spec = importlib.util.spec_from_file_location("_growth_cli_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(tmp_path, cap=None):
    return argparse.Namespace(
        aligned_h5ad=str(tmp_path / "aligned.h5ad"),
        model_dir=str(tmp_path / "training"),
        output_dir=str(tmp_path / "out"),
        device="cuda",
        random_seed=42,
        annotation_key="Annotation",
        max_cells_per_timepoint=cap,
    )


def test_figure5e_contract(monkeypatch, tmp_path):
    calls = []
    cli = _load(monkeypatch, calls)
    cli.run(_args(tmp_path))
    config, kwargs = calls[0]
    assert config.kwargs["model_format"] == "current"
    assert config.kwargs["spatial_warp_to_observed_piecewise"] is False
    assert config.kwargs["run_communication"] is False
    assert config.kwargs["run_3d"] is False
    assert kwargs["max_cells_per_timepoint"] is None


def test_rejects_nonpositive_cap(monkeypatch, tmp_path):
    cli = _load(monkeypatch, [])
    with pytest.raises(ValueError, match="positive"):
        cli.run(_args(tmp_path, cap=0))
