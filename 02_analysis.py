from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms
    from MDAnalysis.analysis.base import AnalysisFromFunction
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit(
        "MDAnalysis is required for structural analysis. "
        "Install it with `pip install MDAnalysis` and retry."
    ) from exc


FIGURE_DPI = 300
R_GAS_CONSTANT = 8.314462618  # J / (mol * K)
BAR_NM3_TO_KJMOL = 0.06022140857  # (bar * nm^3) to kJ/mol


@dataclass(frozen=True)
class AnalysisInputs:
    topology: Path
    trajectory: Path
    reference: Path
    log: Path
    step_size_ps: float
    dcd_interval: int
    temperature_kelvin: float
    pressure_bar: float
    output_dir: Path


def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' was not found.")
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(
    override: Optional[Path],
    config_paths: dict[str, Path],
    key: str,
    description: str,
) -> Path:
    if override is not None:
        candidate = override
    else:
        if key not in config_paths:
            raise ValueError(
                f"{description} not provided and '{key}' missing from config paths."
            )
        candidate = config_paths[key]
    if not candidate.exists():
        raise FileNotFoundError(f"{description} file '{candidate}' was not found.")
    return candidate


def resolve_inputs(args: argparse.Namespace) -> AnalysisInputs:
    config_path: Path = args.config.resolve()
    config = load_yaml_config(config_path)

    try:
        raw_paths = config["paths"]
    except KeyError as exc:
        raise KeyError(
            f"'paths' section missing from config '{config_path}'."
        ) from exc

    config_base = config_path.parent
    config_paths: dict[str, Path] = {}
    for key, value in raw_paths.items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (config_base / candidate).resolve()
        config_paths[key] = candidate

    topology = resolve_path(
        args.topology, config_paths, "topology", "Topology"
    )
    trajectory = resolve_path(
        args.trajectory, config_paths, "trajectory", "Trajectory"
    )
    reference = resolve_path(
        args.reference, config_paths, "pdb", "Reference structure"
    )
    log_file = resolve_path(
        args.log, config_paths, "log", "Reporter log"
    )

    try:
        thermodynamics = config["thermodynamics"]
    except KeyError as exc:
        raise KeyError(
            f"'thermodynamics' section missing from config '{config_path}'."
        ) from exc
    try:
        temperature_kelvin = float(thermodynamics["temperature"])
        pressure_bar = float(thermodynamics["pressure"])
        step_size_ps = float(thermodynamics["step_size"])
    except KeyError as exc:
        raise KeyError(
            "Thermodynamics section must define 'temperature', 'pressure', and 'step_size'."
        ) from exc

    if temperature_kelvin <= 0.0 or step_size_ps <= 0.0:
        raise ValueError("Temperature and step_size must be positive.")

    try:
        reporting = config["reporting"]
    except KeyError as exc:
        raise KeyError(
            f"'reporting' section missing from config '{config_path}'."
        ) from exc
    try:
        dcd_interval = int(
            args.dcd_interval
            if args.dcd_interval is not None
            else reporting["dcd_interval"]
        )
    except KeyError as exc:
        raise KeyError(
            "Reporting section must define 'dcd_interval', or override via CLI."
        ) from exc

    if dcd_interval <= 0:
        raise ValueError("dcd_interval must be a positive integer.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return AnalysisInputs(
        topology=topology,
        trajectory=trajectory,
        reference=reference,
        log=log_file,
        step_size_ps=step_size_ps,
        dcd_interval=dcd_interval,
        temperature_kelvin=temperature_kelvin,
        pressure_bar=pressure_bar,
        output_dir=output_dir,
    )


def load_state_data(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",", comments="#")
    except OSError as exc:
        raise RuntimeError(f"Failed to load reporter log '{path}'.") from exc


def compute_time_axis(
    data: np.ndarray,
    step_size_ps: float,
) -> np.ndarray:
    if data.shape[1] > 1:
        time_ps = data[:, 1]
    else:
        steps = data[:, 0]
        time_ps = steps * step_size_ps
    return np.asarray(time_ps, dtype=float) / 1000.0


def plot_columns(
    time_ns: np.ndarray,
    data: np.ndarray,
    columns: list[tuple[str, int, str]],
    output_path: Path,
) -> None:
    available = [col for col in columns if data.shape[1] > col[1]]
    if not available:
        return
    n_rows = len(available)
    fig, axes = plt.subplots(
        n_rows,
        1,
        sharex=True,
        figsize=(8.0, 2.6 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]
    for ax, (label, idx, ylabel) in zip(axes, available):
        ax.plot(time_ns, data[:, idx], lw=1.5)
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (ns)")
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def compute_enthalpy_series(
    data: np.ndarray,
    pressure_bar: float,
) -> np.ndarray:
    required = {"potential": 2, "kinetic": 3, "volume": 6}
    for name, idx in required.items():
        if data.shape[1] <= idx:
            raise ValueError(
                f"Column '{name}' (index {idx}) missing from log data."
            )
    potential = data[:, required["potential"]]
    kinetic = data[:, required["kinetic"]]
    volume = data[:, required["volume"]]
    pv_term = pressure_bar * volume * BAR_NM3_TO_KJMOL
    return potential + kinetic + pv_term


def block_statistics(
    series: np.ndarray,
    block_size: Optional[int],
) -> tuple[float, float, int]:
    if block_size is None or block_size <= 1:
        variance = series.var(ddof=1) if series.size > 1 else 0.0
        return series.mean(), variance, series.size
    blocks = series.size // block_size
    if blocks < 2:
        raise ValueError("Block size too large; need at least two blocks.")
    trimmed = blocks * block_size
    reshaped = series[:trimmed].reshape(blocks, block_size)
    block_means = reshaped.mean(axis=1)
    mean = block_means.mean()
    variance = block_means.var(ddof=1) * block_size
    return mean, variance, blocks


def compute_heat_capacity(
    data: np.ndarray,
    pressure_bar: float,
    temperature_kelvin: float,
    block_size: Optional[int],
) -> tuple[float, float]:
    if temperature_kelvin <= 0.0:
        raise ValueError("Temperature must be positive.")
    enthalpy = compute_enthalpy_series(data, pressure_bar)
    enthalpy_jmol = enthalpy * 1000.0
    mean, variance, samples = block_statistics(enthalpy_jmol, block_size)
    cp_jmolk = variance / (R_GAS_CONSTANT * temperature_kelvin ** 2)
    cp_kjmolk = cp_jmolk / 1000.0
    stderr = 0.0
    if block_size and block_size > 1 and samples > 1:
        stderr = cp_kjmolk * (2.0 / (samples - 1)) ** 0.5
    return mean / 1000.0, cp_kjmolk, stderr


def running_heat_capacity(
    data: np.ndarray,
    pressure_bar: float,
    temperature_kelvin: float,
) -> np.ndarray:
    enthalpy = compute_enthalpy_series(data, pressure_bar) * 1000.0
    cp_series = np.full(enthalpy.shape, np.nan, dtype=float)
    mean = 0.0
    m2 = 0.0
    for idx, value in enumerate(enthalpy, start=1):
        delta = value - mean
        mean += delta / idx
        m2 += delta * (value - mean)
        if idx >= 2:
            variance = m2 / (idx - 1)
            cp_jmolk = variance / (R_GAS_CONSTANT * temperature_kelvin ** 2)
            cp_series[idx - 1] = cp_jmolk / 1000.0
    return cp_series


def save_heat_capacity_plot(
    time_ns: np.ndarray,
    cp_series: np.ndarray,
    cp_estimate: float,
    cp_stderr: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    ax.plot(time_ns, cp_series, lw=1.4, label="Running Cp")
    ax.axhline(cp_estimate, color="tab:red", linestyle="--", label="Final Cp")
    if cp_stderr > 0.0:
        ax.fill_between(
            time_ns,
            cp_estimate - cp_stderr,
            cp_estimate + cp_stderr,
            color="tab:red",
            alpha=0.15,
            label="Cp ± stderr",
        )
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Cp (kJ/mol/K)")
    ax.set_title("Running isobaric heat capacity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def aggregate_rmsf(
    atom_group: mda.core.groups.AtomGroup,
    per_atom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if atom_group.n_atoms != per_atom.size:
        raise ValueError("Atom group size and RMSF array length mismatch.")
    residues: dict[int, list[float]] = {}
    for resid, value in zip(atom_group.resids, per_atom):
        residues.setdefault(int(resid), []).append(float(value))
    residue_ids = np.array(sorted(residues))
    residue_rmsf = np.array(
        [np.mean(residues[resid]) for resid in residue_ids]
    )
    return residue_ids, residue_rmsf


def build_frame_time_axis(
    n_frames: int,
    step_size_ps: float,
    interval: int,
) -> np.ndarray:
    indices = np.arange(n_frames, dtype=float)
    return indices * step_size_ps * interval / 1000.0


def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    ax.plot(x, y, lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def run_structural_analyses(
    inputs: AnalysisInputs,
    rmsd_selection: str,
    rmsf_selection: str,
    rg_selection: str,
) -> None:
    universe = mda.Universe(str(inputs.topology), str(inputs.trajectory))
    reference = mda.Universe(str(inputs.reference))

    align.AlignTraj(
        universe,
        reference,
        select=rmsd_selection,
        in_memory=True,
    ).run()

    rmsd_analysis = rms.RMSD(
        universe,
        reference,
        select=rmsd_selection,
        ref_frame=0,
    ).run()
    rmsd_values = np.asarray(rmsd_analysis.results.rmsd[:, 2])
    rmsd_time = build_frame_time_axis(
        rmsd_values.size, inputs.step_size_ps, inputs.dcd_interval
    )
    save_line_plot(
        rmsd_time,
        rmsd_values,
        "Time (ns)",
        "RMSD (Å)",
        inputs.output_dir / "rmsd.png",
        title=f"RMSD – {rmsd_selection}",
    )

    rmsf_atoms = universe.select_atoms(rmsf_selection)
    if rmsf_atoms.n_atoms == 0:
        raise ValueError(
            f"RMSF selection '{rmsf_selection}' did not match any atoms."
        )
    rmsf_analysis = rms.RMSF(rmsf_atoms).run()
    resid_ids, resid_rmsf = aggregate_rmsf(
        rmsf_atoms,
        np.asarray(rmsf_analysis.results.rmsf),
    )
    save_line_plot(
        resid_ids,
        resid_rmsf,
        "Residue id",
        "RMSF (Å)",
        inputs.output_dir / "rmsf.png",
        title=f"RMSF – {rmsf_selection}",
    )

    rg_atoms = universe.select_atoms(rg_selection)
    if rg_atoms.n_atoms == 0:
        raise ValueError(
            f"Radius of gyration selection '{rg_selection}' did not match any atoms."
        )
    rg_analysis = AnalysisFromFunction(
        lambda ag: ag.radius_of_gyration(), rg_atoms
    ).run()
    rg_values = np.asarray(rg_analysis.results.timeseries)
    rg_time = build_frame_time_axis(
        rg_values.size, inputs.step_size_ps, inputs.dcd_interval
    )
    save_line_plot(
        rg_time,
        rg_values,
        "Time (ns)",
        "Radius of gyration (Å)",
        inputs.output_dir / "radius_of_gyration.png",
        title=f"Radius of gyration – {rg_selection}",
    )


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-colorblind")


def main(args: argparse.Namespace) -> None:
    configure_matplotlib()
    inputs = resolve_inputs(args)

    data = load_state_data(inputs.log)
    time_ns = compute_time_axis(data, inputs.step_size_ps)

    plot_columns(
        time_ns,
        data,
        [
            ("Potential energy", 2, "kJ/mol"),
            ("Kinetic energy", 3, "kJ/mol"),
            ("Total energy", 4, "kJ/mol"),
        ],
        inputs.output_dir / "energies.png",
    )

    plot_columns(
        time_ns,
        data,
        [
            ("Temperature", 5, "K"),
            ("Box volume", 6, "nm^3"),
            ("Density", 7, "g/mL"),
        ],
        inputs.output_dir / "thermodynamics.png",
    )

    _, cp_value, cp_stderr = compute_heat_capacity(
        data,
        inputs.pressure_bar,
        inputs.temperature_kelvin,
        args.block_size,
    )
    cp_running = running_heat_capacity(
        data,
        inputs.pressure_bar,
        inputs.temperature_kelvin,
    )
    save_heat_capacity_plot(
        time_ns,
        cp_running,
        cp_value,
        cp_stderr,
        inputs.output_dir / "heat_capacity.png",
    )
    run_structural_analyses(
        inputs,
        args.selection,
        args.selection_rmsf,
        args.selection_rg,
    )
    print(
        f"Analysis complete. Figures saved under '{inputs.output_dir}'."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive MD trajectory analysis with publication-ready figures."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration (default: config.yaml).",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        help="Topology file path (overrides config paths.topology).",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Trajectory file path (overrides config paths.trajectory).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference structure path for alignment (defaults to paths.pdb).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Reporter log path (overrides config paths.log).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory where figures will be saved (default: analysis).",
    )
    parser.add_argument(
        "--selection",
        default="protein and name CA",
        help="Selection used for RMSD alignment (default: protein and name CA).",
    )
    parser.add_argument(
        "--selection-rmsf",
        default="protein and name CA",
        help="Selection used for RMSF averaging (default: protein and name CA).",
    )
    parser.add_argument(
        "--selection-rg",
        default="protein",
        help="Selection used for radius of gyration (default: protein).",
    )
    parser.add_argument(
        "--dcd-interval",
        type=int,
        help="Override the reporting interval used for trajectory frames.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        help="Optional block size (frames) for Cp uncertainty estimation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
