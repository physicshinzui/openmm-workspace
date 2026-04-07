from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reimage and center an MD trajectory with MDTraj."
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="Input trajectory file, for example traj.dcd.",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        required=True,
        help="Topology/reference file matching the trajectory atom order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output trajectory path. Defaults to <trajectory_stem>_centered.dcd.",
    )
    parser.add_argument(
        "--output-pdb",
        type=Path,
        help="Optional output PDB path for the first centered frame.",
    )
    parser.add_argument(
        "--anchor-selection",
        type=str,
        default="protein",
        help=(
            "MDTraj atom selection used to define the anchor molecule for imaging "
            "(default: protein). If it matches no atoms, the largest molecule is used."
        ),
    )
    return parser.parse_args()


def default_output_path(trajectory_path: Path) -> Path:
    return trajectory_path.with_name(f"{trajectory_path.stem}_centered.dcd")


def resolve_anchor_molecules(
    topology, selection: Optional[str]
) -> tuple[Optional[list[list[object]]], str]:
    if selection is None or not selection.strip():
        return None, "Using MDTraj default anchor molecules."

    anchor_indices = topology.select(selection)
    if len(anchor_indices) == 0:
        return (
            None,
            f"Selection '{selection}' matched no atoms; using MDTraj default anchor molecules.",
        )

    anchor_atoms = [topology.atom(int(index)) for index in anchor_indices]

    return [anchor_atoms], (
        f"Using selection '{selection}' as the anchor molecule "
        f"({len(anchor_indices)} atoms)."
    )


def main() -> None:
    args = parse_args()

    try:
        import mdtraj as md
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(
            "Missing optional dependency MDTraj. "
            "Install it with `conda install -c conda-forge mdtraj` and try again."
        ) from exc

    output_path = args.output or default_output_path(args.trajectory)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = md.load(str(args.trajectory), top=str(args.topology))
    anchor_molecules, anchor_message = resolve_anchor_molecules(
        trajectory.topology, args.anchor_selection
    )

    trajectory.image_molecules(inplace=True, anchor_molecules=anchor_molecules)
    trajectory.center_coordinates()
    trajectory.save_dcd(str(output_path))

    print(anchor_message)
    print(f"Wrote centered trajectory to {output_path}")

    if args.output_pdb is not None:
        args.output_pdb.parent.mkdir(parents=True, exist_ok=True)
        trajectory[0].save_pdb(str(args.output_pdb))
        print(f"Wrote centered first frame to {args.output_pdb}")


if __name__ == "__main__":
    main()
