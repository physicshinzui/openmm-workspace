from __future__ import annotations

import argparse

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_LOG_PATH = Path("md_log.txt")
DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(config_path: Path) -> dict:
    with config_path.open() as handle:
        return yaml.safe_load(handle) or {}


def load_step_size_ps(config: dict, config_path: Path) -> float:
    try:
        thermodynamics = config["thermodynamics"]
    except KeyError as exc:
        raise KeyError(
            f"'thermodynamics' section not found in config '{config_path}'."
        ) from exc

    try:
        return float(thermodynamics["step_size"])
    except KeyError as exc:
        raise KeyError(
            f"'step_size' missing from thermodynamics in '{config_path}'."
        ) from exc


def load_dcd_interval(config: dict, config_path: Path) -> int:
    try:
        reporting = config["reporting"]
    except KeyError as exc:
        raise KeyError(
            f"'reporting' section not found in config '{config_path}'."
        ) from exc
    try:
        return int(reporting["dcd_interval"])
    except KeyError as exc:
        raise KeyError(
            f"'dcd_interval' missing from reporting in '{config_path}'."
        ) from exc


def load_paths(config: dict, config_path: Path) -> dict[str, Path]:
    try:
        raw_paths = config["paths"]
    except KeyError as exc:
        raise KeyError(
            f"'paths' section not found in config '{config_path}'."
        ) from exc
    return {key: Path(value) for key, value in raw_paths.items()}


def build_time_axis(
    n_frames: int, step_size_ps: float, interval: int
) -> np.ndarray:
    frame_indices = np.arange(n_frames, dtype=float)
    return frame_indices * step_size_ps * interval / 1000.0


def load_log(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", comments="#")


def plot_series(time_ns: np.ndarray, values: np.ndarray, ylabel: str) -> None:
    plt.figure()
    plt.plot(time_ns, values)
    plt.xlabel("Time (ns)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_rmsf(residue_ids: np.ndarray, values: np.ndarray, selection: str) -> None:
    plt.figure()
    plt.plot(residue_ids, values)
    plt.xlabel("Residue id")
    plt.ylabel("RMSF (Å)")
    plt.title(f"RMSF – selection: {selection}")
    plt.tight_layout()
    plt.show()


def compute_structural_metrics(
    topology_path: Path,
    trajectory_path: Path,
    reference_path: Path,
    step_size_ps: float,
    dcd_interval: int,
    rmsd_selection: str,
    rmsf_selection: str,
    rg_selection: str,
) -> None:
    try:
        from MDAnalysis import Universe
        from MDAnalysis.analysis import align, rms
        from MDAnalysis.analysis.base import AnalysisFromFunction
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Structural analysis requires MDAnalysis. "
            "Install it with `pip install MDAnalysis` and retry."
        ) from exc

    for path, description in [
        (topology_path, "topology"),
        (trajectory_path, "trajectory"),
        (reference_path, "reference structure"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{description.capitalize()} file '{path}' was not found."
            )

    universe = Universe(str(topology_path), str(trajectory_path))
    reference = Universe(str(reference_path))

    align.AlignTraj(
        universe,
        reference,
        select=rmsd_selection,
        in_memory=True,
    ).run()

    rmsd_analysis = rms.RMSD(
        universe, reference, select=rmsd_selection, ref_frame=0
    ).run()
    rmsd_values = np.asarray(rmsd_analysis.results.rmsd[:, 2])
    rmsd_time_ns = build_time_axis(
        rmsd_values.shape[0], step_size_ps, dcd_interval
    )
    plot_series(rmsd_time_ns, rmsd_values, f"RMSD (Å) – {rmsd_selection}")

    rmsf_atoms = universe.select_atoms(rmsf_selection)
    if rmsf_atoms.n_atoms == 0:
        raise ValueError(
            f"RMSF selection '{rmsf_selection}' did not match any atoms."
        )
    rmsf_analysis = rms.RMSF(rmsf_atoms).run()
    rmsf_values = np.asarray(rmsf_analysis.results.rmsf)
    resid_lookup: dict[int, list[float]] = {}
    for resid, value in zip(rmsf_atoms.resids, rmsf_values):
        resid_lookup.setdefault(int(resid), []).append(float(value))
    residue_ids = np.array(sorted(resid_lookup))
    residue_rmsf = np.array(
        [np.mean(resid_lookup[resid]) for resid in residue_ids]
    )
    plot_rmsf(residue_ids, residue_rmsf, rmsf_selection)

    rg_atoms = universe.select_atoms(rg_selection)
    if rg_atoms.n_atoms == 0:
        raise ValueError(
            f"Radius of gyration selection '{rg_selection}' did not match any atoms."
        )
    rg_analysis = AnalysisFromFunction(
        lambda ag: ag.radius_of_gyration(), rg_atoms
    ).run()
    rg_values = np.asarray(rg_analysis.results.timeseries)
    rg_time_ns = build_time_axis(rg_values.shape[0], step_size_ps, dcd_interval)
    plot_series(rg_time_ns, rg_values, f"Radius of gyration (Å) – {rg_selection}")


def main(
    log_path: Path,
    config_path: Path,
    topology: Optional[Path],
    trajectory: Optional[Path],
    reference: Optional[Path],
    rmsd_selection: str,
    rmsf_selection: str,
    rg_selection: str,
    override_dcd_interval: Optional[int],
) -> None:
    config = load_config(config_path)
    step_size_ps = load_step_size_ps(config, config_path)
    dcd_interval = (
        override_dcd_interval
        if override_dcd_interval is not None
        else load_dcd_interval(config, config_path)
    )
    paths = load_paths(config, config_path)

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

    topology_path = topology or paths.get("topology")
    trajectory_path = trajectory or paths.get("trajectory")
    reference_path = reference or paths.get("pdb") or topology_path

    if topology_path is None or trajectory_path is None or reference_path is None:
        raise ValueError(
            "Unable to resolve topology/trajectory/reference paths. "
            "Specify them explicitly or ensure they are present under 'paths' in the config."
        )

    compute_structural_metrics(
        topology_path,
        trajectory_path,
        reference_path,
        step_size_ps,
        dcd_interval,
        rmsd_selection,
        rmsf_selection,
        rg_selection,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MD observables versus time (ns) and structural metrics."
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
        help=(
            f"Path to YAML config containing thermodynamic parameters "
            f"(default: {DEFAULT_CONFIG_PATH})"
        ),
    )
    parser.add_argument(
        "--topology",
        type=Path,
        help="Topology file path (defaults to paths.topology in the config).",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Trajectory file path (defaults to paths.trajectory in the config).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference structure for RMSD alignment (defaults to paths.pdb or topology).",
    )
    parser.add_argument(
        "--selection",
        default="protein and name CA",
        help="Atom selection for RMSD and alignment (default: 'protein and name CA').",
    )
    parser.add_argument(
        "--selection-rmsf",
        default="protein and name CA",
        help="Atom selection for RMSF (default: 'protein and name CA').",
    )
    parser.add_argument(
        "--selection-rg",
        default="protein",
        help="Atom selection for radius of gyration (default: 'protein').",
    )
    parser.add_argument(
        "--dcd-interval",
        type=int,
        help="Override the trajectory reporting interval used for time conversion.",
    )
    args = parser.parse_args()
    main(
        args.log,
        args.config,
        args.topology,
        args.trajectory,
        args.reference,
        args.selection,
        args.selection_rmsf,
        args.selection_rg,
        args.dcd_interval,
    )
