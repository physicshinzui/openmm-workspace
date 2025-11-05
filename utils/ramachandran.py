from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEGREE_RANGE = (-180.0, 180.0)


def load_config(path: Path) -> dict:
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file '{path}' was not found.")
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def load_paths(config: dict, config_path: Path) -> dict[str, Path]:
    try:
        raw_paths = config["paths"]
    except KeyError:
        return {}
    root = config_path.parent
    resolved: dict[str, Path] = {}
    for key, value in raw_paths.items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        resolved[key] = candidate
    return resolved


def resolve_input_path(
    explicit: Optional[Path],
    config_paths: dict[str, Path],
    key: str,
    description: str,
) -> Path:
    if explicit is not None:
        return explicit
    if key not in config_paths:
        raise ValueError(
            f"{description} not provided and '{key}' missing from config paths."
        )
    return config_paths[key]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse phi/psi angles for a dipeptide and generate a Ramachandran plot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config (default: config.yaml).",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        help="Topology file (PDB/PSF/etc.). Overrides config paths.topology.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Trajectory file (DCD/XT: etc.). Overrides config paths.trajectory.",
    )
    parser.add_argument(
        "--selection",
        default="all",
        help="MDAnalysis selection string that isolates the dipeptide (default: all atoms).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=72,
        help="Number of bins per axis for the 2D histogram (default: 72).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the Ramachandran figure (PNG/SVG/etc.).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to export phi/psi samples as CSV.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the interactive Matplotlib window.",
    )
    return parser.parse_args()


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def compute_phi_psi(
    topology_path: Path,
    trajectory_path: Path,
    selection: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        from MDAnalysis import Universe
        from MDAnalysis.analysis.dihedrals import Ramachandran
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Ramachandran analysis requires MDAnalysis. "
            "Install it with `pip install MDAnalysis` and retry."
        ) from exc

    if not topology_path.exists():
        raise FileNotFoundError(f"Topology file '{topology_path}' was not found.")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file '{trajectory_path}' was not found.")

    universe = Universe(str(topology_path), str(trajectory_path))
    atom_group = universe.select_atoms(selection)
    if atom_group.n_atoms == 0:
        raise ValueError(f"Selection '{selection}' did not match any atoms.")

    ramachandran = Ramachandran(atom_group).run()
    angles_rad = np.asarray(ramachandran.results.angles)
    if angles_rad.size == 0:
        raise ValueError(
            "No phi/psi pairs were found. "
            "Confirm that the selection covers consecutive residues with neighbours."
        )

    phi_rad = angles_rad[..., 0]
    psi_rad = angles_rad[..., 1]

    mask = np.isfinite(phi_rad) & np.isfinite(psi_rad)
    phi_deg = wrap_degrees(np.rad2deg(phi_rad[mask]))
    psi_deg = wrap_degrees(np.rad2deg(psi_rad[mask]))

    resids = getattr(ramachandran.results, "resids", None)
    resnames = getattr(ramachandran.results, "resnames", None)
    labels: list[str] = []
    if resids is not None and resnames is not None:
        labels = [f"{name}{resid}" for name, resid in zip(resnames, resids)]
    return phi_deg, psi_deg, labels


def plot_ramachandran(
    phi_deg: np.ndarray,
    psi_deg: np.ndarray,
    bins: int,
    output_path: Optional[Path],
    show_plot: bool,
) -> None:
    plt.figure()
    hist = plt.hist2d(
        phi_deg,
        psi_deg,
        bins=bins,
        range=[DEGREE_RANGE, DEGREE_RANGE],
        cmap="viridis",
        density=True,
    )
    plt.colorbar(hist[3], label="Probability density")
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Psi (degrees)")
    plt.title("Ramachandran plot")
    plt.xlim(*DEGREE_RANGE)
    plt.ylim(*DEGREE_RANGE)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def export_csv(path: Path, phi_deg: np.ndarray, psi_deg: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((phi_deg, psi_deg))
    header = "phi_deg,psi_deg"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    config_paths = load_paths(config, args.config)

    topology = resolve_input_path(
        args.topology, config_paths, "topology", "Topology file"
    )
    trajectory = resolve_input_path(
        args.trajectory, config_paths, "trajectory", "Trajectory file"
    )

    phi_deg, psi_deg, residue_labels = compute_phi_psi(
        topology, trajectory, args.selection
    )

    n_samples = phi_deg.size
    if residue_labels:
        print("Residues included:", ", ".join(residue_labels))
    print(f"Collected {n_samples} phi/psi pairs.")
    print(
        f"Mean phi: {phi_deg.mean():.2f}°, mean psi: {psi_deg.mean():.2f}°, "
        f"std phi: {phi_deg.std(ddof=1):.2f}°, std psi: {psi_deg.std(ddof=1):.2f}°."
    )

    if args.csv is not None:
        export_csv(args.csv, phi_deg, psi_deg)
        print(f"Wrote raw samples to '{args.csv}'.")

    show_plot = not args.no_show
    plot_ramachandran(phi_deg, psi_deg, args.bins, args.output, show_plot)


if __name__ == "__main__":
    main()
