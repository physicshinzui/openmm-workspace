from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from openmm.app import DCDFile, PDBFile

from openmmtools.multistate import MultiStateReporter


def write_dcd_trajectories(
    storage_path: Path,
    topology_path: Path,
    output_dir: Path,
    frame_time_ps: float,
    replica_indices: Sequence[int] | None = None,
) -> None:
    """Export stored sampler states to per-replica DCD trajectories."""
    reporter = MultiStateReporter(str(storage_path), open_mode="r")
    try:
        topology = PDBFile(str(topology_path)).topology
        n_replicas = reporter.n_replicas or 0
        if n_replicas == 0:
            return
        indices = list(replica_indices) if replica_indices is not None else list(range(n_replicas))
        output_dir.mkdir(parents=True, exist_ok=True)

        handles: dict[int, object] = {}
        writers: dict[int, DCDFile] = {}
        try:
            for replica in indices:
                trajectory_path = output_dir / f"replica_{replica}.dcd"
                handle = trajectory_path.open("wb")
                handles[replica] = handle
                writers[replica] = DCDFile(handle, topology, dt=frame_time_ps)

            last_iteration = reporter.read_last_iteration()
            if last_iteration is None:
                return
            for iteration in range(last_iteration + 1):
                try:
                    sampler_states = reporter.read_sampler_states(iteration)
                except Exception:
                    continue
                if not sampler_states:
                    continue
                for replica in indices:
                    if replica >= len(sampler_states):
                        continue
                    sampler_state = sampler_states[replica]
                    if sampler_state is None:
                        continue
                    positions = sampler_state.positions
                    if positions is None:
                        continue
                    writers[replica].writeModel(positions)
        finally:
            for writer in writers.values():
                # DCDFile does not expose a close method; flush by closing handle.
                pass
            for handle in handles.values():
                handle.close()
    finally:
        reporter.close()
