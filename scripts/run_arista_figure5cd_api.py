#!/usr/bin/env python3
"""Regenerate corrected ARISTA Figure 5c/d panels through public APIs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_helpers.arista_api import (  # noqa: E402
    AristaSpatiotemporalConfig,
    assert_package_only_runtime,
    run_arista_direction_correlation_api,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Figure 5c uses the requested observed spatial slice; Figure 5d "
            "uses all observed cells and a 50-D gene-state velocity graph."
        )
    )
    parser.add_argument("--aligned-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-format", choices=("current", "legacy"), default="current")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-timepoint", type=float, default=1.0)
    parser.add_argument("--focus-label", default="reaEGC")
    parser.add_argument("--roi-pad-ratio", type=float, default=0.15)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--max-spatial-cells", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = AristaSpatiotemporalConfig(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        aligned_h5ad=Path(args.aligned_h5ad).expanduser().resolve(),
        model_dir=Path(args.model_dir).expanduser().resolve(),
        model_format=args.model_format,
        random_seed=int(args.random_seed),
        device=args.device,
        run_communication=False,
        run_3d=False,
    )
    result = run_arista_direction_correlation_api(
        config,
        target_timepoint=float(args.target_timepoint),
        focus_label_keyword=args.focus_label,
        pad_ratio=float(args.roi_pad_ratio),
        n_neighbors=int(args.n_neighbors),
        max_cells=args.max_spatial_cells,
    )
    assert_package_only_runtime()
    print(result.manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
