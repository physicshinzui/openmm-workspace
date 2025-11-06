from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from openmmtools.multistate import MultiStateReporter


def _format_quantity(quantity) -> str:
    try:
        value = quantity.value_in_unit(quantity.unit.__class__.standard_unit)
        unit_string = str(quantity.unit)
    except Exception:
        try:
            value = quantity.value_in_unit_system(quantity.unit.system)
            unit_string = str(quantity.unit)
        except Exception:
            return str(quantity)
    return f"{value} {unit_string}"


def _describe_thermodynamic_state(state) -> dict[str, float | str | None]:
    summary: dict[str, float | str | None] = {}
    # Standard attributes
    temperature = getattr(state, "temperature", None)
    pressure = getattr(state, "pressure", None)
    summary["temperature_K"] = (
        float(temperature.value_in_unit(temperature.unit)) if temperature is not None else None
    )
    summary["pressure_bar"] = (
        float(pressure.value_in_unit(pressure.unit)) if pressure is not None else None
    )
    for attr in (
        "lambda_sterics",
        "lambda_electrostatics",
        "lambda_bonds",
        "lambda_angles",
        "lambda_torsions",
    ):
        if hasattr(state, attr):
            summary[attr] = float(getattr(state, attr))
    return summary


def dump_storage_to_text(storage_path: Path, output_path: Path) -> None:
    """Serialize all accessible data from storage.nc into a human-readable text file."""

    reporter = MultiStateReporter(str(storage_path), open_mode="r")
    try:
        last_iteration = reporter.read_last_iteration()
        if last_iteration is None:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            handle.write(f"Storage file: {storage_path}\n")
            handle.write(f"Total iterations: {last_iteration + 1}\n")
            handle.write(f"Number of replicas: {reporter.n_replicas}\n")
            handle.write(f"Number of thermodynamic states: {reporter.n_states}\n")
            handle.write("\nThermodynamic States:\n")

            try:
                thermo_states = reporter.read_thermodynamic_states()
            except Exception:
                thermo_states = []
            for index, state in enumerate(thermo_states):
                summary = _describe_thermodynamic_state(state)
                handle.write(f"  State {index}: " + ", ".join(
                    f"{key}={value}" for key, value in summary.items() if value is not None
                ) + "\n")

            handle.write("\nIterations:\n")
            total_accepted = None
            total_proposed = None

            for iteration in range(last_iteration + 1):
                handle.write(f"\nIteration {iteration}\n")
                try:
                    timestamp = reporter.read_timestamp(iteration)
                    handle.write(f"  Timestamp (ns): {timestamp}\n")
                except Exception:
                    pass

                try:
                    replica_states = reporter.read_replica_thermodynamic_states(iteration)
                    if replica_states is not None:
                        handle.write(
                            "  Replica state indices: "
                            + " ".join(str(int(idx)) for idx in replica_states)
                            + "\n"
                        )
                except Exception:
                    pass

                try:
                    energies, _, _ = reporter.read_energies(iteration)
                except Exception:
                    energies = None

                if energies is not None:
                    handle.write("  Reduced potentials (kT):\n")
                    for replica_index, row in enumerate(energies):
                        row_values = " ".join(
                            f"{value: .6f}" if math.isfinite(value) else "nan"
                            for value in row
                        )
                        handle.write(f"    replica {replica_index}: {row_values}\n")

                try:
                    accepted, proposed = reporter.read_mixing_statistics(iteration)
                except Exception:
                    accepted = proposed = None
                if accepted is not None and proposed is not None:
                    if total_accepted is None:
                        total_accepted = np.array(accepted, dtype=np.int64)
                        total_proposed = np.array(proposed, dtype=np.int64)
                    else:
                        total_accepted += accepted
                        total_proposed += proposed

                    handle.write("  Mixing statistics (accepted/proposed):\n")
                    for i in range(accepted.shape[0]):
                        entries = []
                        for j in range(accepted.shape[1]):
                            acc = int(accepted[i, j])
                            prop = int(proposed[i, j])
                            entries.append(f"{acc}/{prop}")
                        handle.write(f"    state {i} -> {entries}\n")

                    with np.errstate(divide="ignore", invalid="ignore"):
                        accept_rates = np.where(
                            proposed > 0, accepted.astype(float) / proposed, np.nan
                        )
                    handle.write("  Acceptance ratios:\n")
                    for i in range(accept_rates.shape[0]):
                        entries = []
                        for j in range(accept_rates.shape[1]):
                            value = accept_rates[i, j]
                            entries.append(
                                f"{value:.3f}" if np.isfinite(value) else "nan"
                            )
                        handle.write(f"    state {i} -> {entries}\n")

            if total_accepted is not None and total_proposed is not None:
                handle.write("\nAggregated mixing statistics:\n")
                with np.errstate(divide="ignore", invalid="ignore"):
                    aggregated = np.where(
                        total_proposed > 0,
                        total_accepted.astype(float) / total_proposed,
                        np.nan,
                    )
                for i in range(aggregated.shape[0]):
                    entries = []
                    for j in range(aggregated.shape[1]):
                        value = aggregated[i, j]
                        entries.append(f"{value:.3f}" if np.isfinite(value) else "nan")
                    handle.write(f"  state {i} -> {entries}\n")

                finite_values = aggregated[np.isfinite(aggregated)]
                if finite_values.size:
                    average_acceptance = float(np.mean(finite_values))
                else:
                    average_acceptance = float("nan")
                handle.write(f"\nMean acceptance (finite entries only): {average_acceptance:.3f}\n")

    finally:
        reporter.close()
