from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("data")
DEFAULT_TRAJECTORY_NAME = "traj.dcd"
DEFAULT_TOPOLOGY_NAME = "topology.pdb"
DEFAULT_SELECTION = "protein"


@dataclass(frozen=True)
class ExtractionJob:
    topology: Path
    trajectory: Path
    output_trajectory: Path
    output_topology: Path

    @property
    def label(self) -> str:
        try:
            return self.trajectory.parents[2].name
        except IndexError:
            return self.trajectory.stem


@dataclass(frozen=True)
class JobResult:
    job: ExtractionJob
    input_atoms: int
    output_atoms: int
    input_frames: int
    output_frames: int
    input_size: int
    output_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a protein-only DCD trajectory and matching PDB topology."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--root",
        type=Path,
        help="Process every matching DCD below this directory.",
    )
    source.add_argument(
        "--trajectory",
        type=Path,
        help="Process one trajectory. Requires --topology and --output.",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        help="Topology matching --trajectory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Protein-only DCD output path when processing one trajectory.",
    )
    parser.add_argument(
        "--output-topology",
        type=Path,
        help=(
            "Protein-only PDB topology output path when processing one trajectory "
            "(default: <output_stem>.pdb)."
        ),
    )
    parser.add_argument(
        "--trajectory-name",
        default=DEFAULT_TRAJECTORY_NAME,
        help=f"Input filename used with --root (default: {DEFAULT_TRAJECTORY_NAME}).",
    )
    parser.add_argument(
        "--topology-name",
        default=DEFAULT_TOPOLOGY_NAME,
        help=f"Topology filename used with --root (default: {DEFAULT_TOPOLOGY_NAME}).",
    )
    parser.add_argument(
        "--output-name",
        default="traj_protein.dcd",
        help="DCD output filename used with --root (default: traj_protein.dcd).",
    )
    parser.add_argument(
        "--output-topology-name",
        default="topology_protein.pdb",
        help="PDB output filename used with --root (default: topology_protein.pdb).",
    )
    parser.add_argument(
        "--selection",
        default=DEFAULT_SELECTION,
        help=f"MDAnalysis atom selection to extract (default: {DEFAULT_SELECTION}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned jobs without opening or writing trajectories.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.trajectory is not None:
        if args.topology is None or args.output is None:
            raise ValueError("--trajectory requires both --topology and --output")
        if args.output_name != "traj_protein.dcd":
            raise ValueError("--output-name can only be used with --root")
        if args.output_topology_name != "topology_protein.pdb":
            raise ValueError("--output-topology-name can only be used with --root")
    elif args.topology is not None or args.output is not None:
        raise ValueError("--topology and --output can only be used with --trajectory")
    elif args.output_topology is not None:
        raise ValueError("--output-topology can only be used with --trajectory")


def default_output_topology(output: Path) -> Path:
    return output.with_suffix(".pdb")


def build_jobs(args: argparse.Namespace) -> list[ExtractionJob]:
    if args.trajectory is not None:
        return [
            ExtractionJob(
                topology=args.topology,
                trajectory=args.trajectory,
                output_trajectory=args.output,
                output_topology=args.output_topology
                or default_output_topology(args.output),
            )
        ]

    trajectories = sorted(args.root.rglob(args.trajectory_name))
    return [
        ExtractionJob(
            topology=trajectory.with_name(args.topology_name),
            trajectory=trajectory,
            output_trajectory=trajectory.with_name(args.output_name),
            output_topology=trajectory.with_name(args.output_topology_name),
        )
        for trajectory in trajectories
    ]


def validate_job_paths(job: ExtractionJob, overwrite: bool) -> None:
    if not job.trajectory.is_file():
        raise ValueError(f"input trajectory does not exist: {job.trajectory}")
    if not job.topology.is_file():
        raise ValueError(f"topology does not exist: {job.topology}")
    if job.trajectory.resolve() == job.output_trajectory.resolve():
        raise ValueError("output trajectory must not overwrite the input trajectory")
    if job.topology.resolve() == job.output_topology.resolve():
        raise ValueError("output topology must not overwrite the input topology")
    for output in (job.output_trajectory, job.output_topology):
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}")


def partial_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.partial{output.suffix}")


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def extract_protein_trajectory(
    job: ExtractionJob, selection: str, overwrite: bool
) -> JobResult:
    validate_job_paths(job, overwrite)

    import numpy as np
    import MDAnalysis as mda
    from MDAnalysis.coordinates.DCD import DCDWriter
    from tqdm import tqdm

    job.output_trajectory.parent.mkdir(parents=True, exist_ok=True)
    job.output_topology.parent.mkdir(parents=True, exist_ok=True)
    temporary_trajectory = partial_path(job.output_trajectory)
    temporary_topology = partial_path(job.output_topology)
    for temporary in (temporary_trajectory, temporary_topology):
        if temporary.exists():
            temporary.unlink()

    universe = mda.Universe(str(job.topology), str(job.trajectory))
    selected = universe.select_atoms(selection)
    if selected.n_atoms == 0:
        raise ValueError(f"selection matched no atoms: {selection!r}")

    input_frames = len(universe.trajectory)
    first_positions = None
    last_positions = None
    first_dimensions = None
    last_dimensions = None

    try:
        with DCDWriter(str(temporary_trajectory), n_atoms=selected.n_atoms) as writer:
            for timestep in tqdm(universe.trajectory, desc=job.label, unit="frame"):
                positions = selected.positions.copy()
                dimensions = timestep.dimensions
                dimensions = None if dimensions is None else dimensions.copy()
                if first_positions is None:
                    first_positions = positions
                    first_dimensions = dimensions
                last_positions = positions
                last_dimensions = dimensions
                writer.write(selected)

        selected.write(str(temporary_topology))

        verification = mda.Universe(
            str(temporary_topology), str(temporary_trajectory), format="DCD"
        )
        if verification.atoms.n_atoms != selected.n_atoms:
            raise RuntimeError(
                "verification failed: output and selected atom counts differ"
            )
        if len(verification.trajectory) != input_frames:
            raise RuntimeError(
                "verification failed: "
                f"expected {input_frames} frames, found {len(verification.trajectory)}"
            )

        verification.trajectory[0]
        if not np.allclose(verification.atoms.positions, first_positions, atol=1e-3):
            raise RuntimeError("verification failed: first frame coordinates differ")
        if (verification.dimensions is None) != (first_dimensions is None) or (
            first_dimensions is not None
            and not np.allclose(verification.dimensions, first_dimensions, atol=1e-3)
        ):
            raise RuntimeError("verification failed: first frame box dimensions differ")

        verification.trajectory[-1]
        if not np.allclose(verification.atoms.positions, last_positions, atol=1e-3):
            raise RuntimeError("verification failed: last frame coordinates differ")
        if (verification.dimensions is None) != (last_dimensions is None) or (
            last_dimensions is not None
            and not np.allclose(verification.dimensions, last_dimensions, atol=1e-3)
        ):
            raise RuntimeError("verification failed: last frame box dimensions differ")

        temporary_trajectory.replace(job.output_trajectory)
        temporary_topology.replace(job.output_topology)
    except Exception:
        for temporary in (temporary_trajectory, temporary_topology):
            if temporary.exists():
                temporary.unlink()
        raise

    return JobResult(
        job=job,
        input_atoms=universe.atoms.n_atoms,
        output_atoms=selected.n_atoms,
        input_frames=input_frames,
        output_frames=input_frames,
        input_size=job.trajectory.stat().st_size,
        output_size=job.output_trajectory.stat().st_size,
    )


def print_dry_run(jobs: Iterable[ExtractionJob], args: argparse.Namespace) -> None:
    for job in jobs:
        status = "overwrite" if args.overwrite else "write"
        existing = [
            str(output)
            for output in (job.output_trajectory, job.output_topology)
            if output.exists()
        ]
        if existing and not args.overwrite:
            status = f"skip: output exists: {', '.join(existing)}"
        print(
            f"{job.label}: {status}\n"
            f"  topology:          {job.topology}\n"
            f"  trajectory:        {job.trajectory}\n"
            f"  output trajectory: {job.output_trajectory}\n"
            f"  output topology:   {job.output_topology}"
        )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        jobs = build_jobs(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not jobs:
        print("error: no input trajectories found", file=sys.stderr)
        return 1

    if args.dry_run:
        print_dry_run(jobs, args)
        print(f"Planned {len(jobs)} job(s): selection={args.selection!r}")
        return 0

    results: list[JobResult] = []
    failures: list[tuple[ExtractionJob, Exception]] = []
    for job in jobs:
        try:
            result = extract_protein_trajectory(
                job=job,
                selection=args.selection,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failures.append((job, exc))
            print(f"{job.label}: FAILED: {exc}", file=sys.stderr)
            continue

        results.append(result)
        print(
            f"{job.label}: {result.input_atoms} -> {result.output_atoms} atoms, "
            f"{result.input_frames} frames, "
            f"{human_size(result.input_size)} -> {human_size(result.output_size)}, OK"
        )

    total_size = sum(result.output_size for result in results)
    print(
        f"Completed: {len(results)} succeeded, {len(failures)} failed, "
        f"{human_size(total_size)} written"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
