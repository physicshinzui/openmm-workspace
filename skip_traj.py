from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("data/amber14sb-tip3pfb/")
DEFAULT_TRAJECTORY_NAME = "traj.dcd"
DEFAULT_TOPOLOGY_NAME = "topology.pdb"


@dataclass(frozen=True)
class TrajectoryJob:
    topology: Path
    trajectory: Path
    output: Path

    @property
    def label(self) -> str:
        try:
            return self.trajectory.parents[2].name
        except IndexError:
            return self.trajectory.stem


@dataclass(frozen=True)
class JobResult:
    job: TrajectoryJob
    input_frames: int
    output_frames: int
    input_size: int
    output_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin one DCD trajectory or every traj.dcd below a root directory."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--root",
        type=Path,
        help="Process every */output/traj.dcd below this directory.",
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
        help="Output path when processing one trajectory.",
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
        help="Output filename used with --root (default: traj_stride<stride>.dcd).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First input frame to retain (default: 0).",
    )
    parser.add_argument(
        "--stop",
        type=int,
        help="Exclusive input frame at which to stop (default: end of trajectory).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Retain every Nth input frame (default: 100).",
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
    if args.stride <= 0:
        raise ValueError("--stride must be greater than zero")
    if args.start < 0:
        raise ValueError("--start must be zero or greater")
    if args.stop is not None and args.stop < 0:
        raise ValueError("--stop must be zero or greater")

    if args.trajectory is not None:
        if args.topology is None or args.output is None:
            raise ValueError("--trajectory requires both --topology and --output")
        if args.output_name is not None:
            raise ValueError("--output-name can only be used with --root")
    elif args.topology is not None or args.output is not None:
        raise ValueError("--topology and --output can only be used with --trajectory")


def build_jobs(args: argparse.Namespace) -> list[TrajectoryJob]:
    if args.trajectory is not None:
        return [
            TrajectoryJob(
                topology=args.topology,
                trajectory=args.trajectory,
                output=args.output,
            )
        ]

    output_name = args.output_name or f"traj_stride{args.stride}.dcd"
    trajectories = sorted(args.root.rglob(args.trajectory_name))
    return [
        TrajectoryJob(
            topology=trajectory.with_name(args.topology_name),
            trajectory=trajectory,
            output=trajectory.with_name(output_name),
        )
        for trajectory in trajectories
    ]


def validate_job_paths(job: TrajectoryJob, overwrite: bool) -> None:
    if not job.trajectory.is_file():
        raise ValueError(f"input trajectory does not exist: {job.trajectory}")
    if not job.topology.is_file():
        raise ValueError(f"topology does not exist: {job.topology}")
    if job.trajectory.resolve() == job.output.resolve():
        raise ValueError("output must not overwrite the input trajectory")
    if job.output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {job.output}")


def partial_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.partial")


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def selected_frame_indices(
    frame_count: int, start: int, stop: int | None, stride: int
) -> range:
    normalized_start, normalized_stop, normalized_stride = slice(
        start, stop, stride
    ).indices(frame_count)
    return range(normalized_start, normalized_stop, normalized_stride)


def thin_trajectory(
    job: TrajectoryJob, start: int, stop: int | None, stride: int, overwrite: bool
) -> JobResult:
    validate_job_paths(job, overwrite)

    import numpy as np
    import MDAnalysis as mda
    from MDAnalysis.coordinates.DCD import DCDWriter
    from tqdm import tqdm

    job.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = partial_path(job.output)
    if temporary.exists():
        temporary.unlink()

    universe = mda.Universe(str(job.topology), str(job.trajectory))
    input_frames = len(universe.trajectory)
    indices = selected_frame_indices(input_frames, start, stop, stride)
    output_frames = len(indices)
    if output_frames == 0:
        raise ValueError("frame selection is empty")

    input_dt = float(universe.trajectory.dt)
    writer_dt = input_dt * stride if math.isfinite(input_dt) else 1.0
    first_positions = None
    last_positions = None
    first_dimensions = None
    last_dimensions = None

    try:
        with DCDWriter(
            str(temporary),
            n_atoms=universe.atoms.n_atoms,
            dt=writer_dt,
        ) as writer:
            for frame_index in tqdm(indices, desc=job.label, unit="frame"):
                universe.trajectory[frame_index]
                positions = universe.atoms.positions.copy()
                dimensions = universe.dimensions
                dimensions = None if dimensions is None else dimensions.copy()
                if first_positions is None:
                    first_positions = positions
                    first_dimensions = dimensions
                last_positions = positions
                last_dimensions = dimensions
                writer.write(universe.atoms)

        verification = mda.Universe(str(job.topology), str(temporary), format="DCD")
        if verification.atoms.n_atoms != universe.atoms.n_atoms:
            raise RuntimeError(
                "verification failed: output and input atom counts differ"
            )
        if len(verification.trajectory) != output_frames:
            raise RuntimeError(
                "verification failed: "
                f"expected {output_frames} frames, found {len(verification.trajectory)}"
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

        temporary.replace(job.output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return JobResult(
        job=job,
        input_frames=input_frames,
        output_frames=output_frames,
        input_size=job.trajectory.stat().st_size,
        output_size=job.output.stat().st_size,
    )


def print_dry_run(jobs: Iterable[TrajectoryJob], args: argparse.Namespace) -> None:
    for job in jobs:
        status = "overwrite" if job.output.exists() and args.overwrite else "write"
        if job.output.exists() and not args.overwrite:
            status = "skip: output exists"
        print(
            f"{job.label}: {status}\n"
            f"  topology:   {job.topology}\n"
            f"  trajectory: {job.trajectory}\n"
            f"  output:     {job.output}"
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
        print(
            f"Planned {len(jobs)} job(s): start={args.start}, "
            f"stop={args.stop}, stride={args.stride}"
        )
        return 0

    results: list[JobResult] = []
    failures: list[tuple[TrajectoryJob, Exception]] = []
    for job in jobs:
        try:
            result = thin_trajectory(
                job=job,
                start=args.start,
                stop=args.stop,
                stride=args.stride,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failures.append((job, exc))
            print(f"{job.label}: FAILED: {exc}", file=sys.stderr)
            continue

        results.append(result)
        print(
            f"{job.label}: {result.input_frames} -> {result.output_frames} frames, "
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
