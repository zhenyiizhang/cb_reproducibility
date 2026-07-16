#!/usr/bin/env python3
"""Run the package-backed ARISTA t1 velocity-stream workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream_helpers.arista_api import (  # noqa: E402
    AristaVelocityConfig,
    run_arista_velocity_t1_streams,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the ARISTA t1 intrinsic/interaction/full velocity panels "
            "through the installed CytoBridge package."
        )
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--edge-predictor-root", default=None)
    parser.add_argument("--label-color-json", default=None)
    parser.add_argument("--output-name", default="arista_velocity_t1_streams_api")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-timepoint", type=float, default=1.0)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--interaction-threshold", type=float, default=1000.0)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--no-legend", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_arista_velocity_t1_streams(
        AristaVelocityConfig(
            output_name=args.output_name,
            output_dir=args.output_dir,
            workspace_root=args.workspace_root,
            data_csv=args.data_csv,
            model_dir=args.model_dir,
            edge_predictor_root=args.edge_predictor_root,
            label_color_json=args.label_color_json,
            target_timepoint=args.target_timepoint,
            max_cells=args.max_cells,
            random_seed=args.random_seed,
            device=args.device,
            interaction_m=args.interaction_m,
            interaction_threshold=args.interaction_threshold,
            density=args.density,
            show_legend=not args.no_legend,
        )
    )
    payload = {
        key: (
            {name: str(path) for name, path in value.items()}
            if isinstance(value, dict)
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in asdict(result).items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
