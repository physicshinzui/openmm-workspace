from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_LOG_PATH = Path("md_log.txt")
DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_step_size_ps(config_path: Path) -> float:
    with config_path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        thermodynamics = raw["thermodynamics"]
    except KeyError as exc:
        raise KeyError(
            f"'thermodynamics' section not found in config '{config_path}'."
        ) from exc

    try:
        step_size_ps = float(thermodynamics["step_size"])
    except KeyError as exc:
        raise KeyError(
            f"'step_size' missing from thermodynamics in '{config_path}'."
        ) from exc
    return step_size_ps


def load_log(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", comments="#")


def plot_series(time_ns: np.ndarray, values: np.ndarray, ylabel: str) -> None:
    plt.plot(time_ns, values)
    plt.xlabel("Time (ns)")
    plt.ylabel(ylabel)
    plt.show()


def main(log_path: Path, config_path: Path) -> None:
    step_size_ps = load_step_size_ps(config_path)
    data = load_log(log_path)

    step = data[:, 0]

    time_ns = (
        data[:, 1] / 1000.0 if data.shape[1] > 1 else step * step_size_ps / 1000.0
    )

    column_map = [
        ("Potential energy (kJ/mol)", 2),
        ("Kinetic energy (kJ/mol)", 3),
        ("Total energy (kJ/mol)", 4),
        ("Temperature (K)", 5),
        ("Volume (nm^3)", 6),
        ("Density (g/mL)", 7),
        ("Speed (ns/day)", 8),
    ]

    for label, col in column_map:
        if data.shape[1] > col:
            plot_series(time_ns, data[:, col], label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MD observables versus time (ns)."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to StateDataReporter log (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config containing thermodynamic parameters (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    main(args.log, args.config)
