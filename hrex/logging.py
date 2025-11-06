from __future__ import annotations

from pathlib import Path

import numpy as np

from openmmtools.multistate import MultiStateReporter


def write_iteration_log(storage_path: Path, output_path: Path) -> None:
    """Export per-replica reduced potentials to a CSV log."""

    reporter = MultiStateReporter(str(storage_path), open_mode="r")
    try:
        last_iteration = reporter.read_last_iteration()
        if last_iteration is None:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            handle.write("iteration,replica,state_index,reduced_potential\n")

            for iteration in range(last_iteration + 1):
                try:
                    state_indices = reporter.read_replica_thermodynamic_states(iteration)
                    energy_data = reporter.read_energies(iteration)
                except Exception:
                    continue

                if state_indices is None or energy_data is None:
                    continue

                energy_states = energy_data[0]
                if energy_states is None:
                    continue

                state_indices = np.asarray(state_indices)
                for replica_index, state_index in enumerate(state_indices):
                    if replica_index >= energy_states.shape[0]:
                        continue
                    try:
                        reduced_potential = float(energy_states[replica_index, int(state_index)])
                    except Exception:
                        continue
                    handle.write(
                        f"{iteration},{replica_index},{int(state_index)},{reduced_potential:.8f}\n"
                    )
    finally:
        reporter.close()
