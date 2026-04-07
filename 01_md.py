from __future__ import annotations

import argparse
from pathlib import Path

from md_config import DEFAULT_CONFIG_PATH, load_simulation_config, resolve_runtime_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenMM MD simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override checkpoint path (defaults to value from config).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Resume from checkpoint and run production stage only.",
    )
    parser.add_argument(
        "--until",
        type=float,
        help="Target production time in nanoseconds. Overrides production_steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation_config = load_simulation_config(args.config)
    checkpoint_path = (
        resolve_runtime_path(args.checkpoint)
        if args.checkpoint is not None
        else simulation_config.paths.checkpoint_path
    )

    from md_workflow import run_simulation

    run_simulation(
        simulation_config,
        restart=args.restart,
        checkpoint_path=checkpoint_path,
        until_ns=args.until,
    )


if __name__ == "__main__":
    main()
