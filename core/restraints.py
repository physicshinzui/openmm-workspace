from __future__ import annotations

from pathlib import Path
from typing import Sequence

import openmm
from openmm import unit
from openmm.app import Topology


def create_position_restraint_force(
    topology: Topology,
    reference_positions: unit.Quantity,
    force_constant: unit.Quantity,
    atom_indices: Sequence[int],
) -> openmm.CustomExternalForce:
    """Create a harmonic positional restraint for the selected atom indices."""

    energy = "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    restraint = openmm.CustomExternalForce(energy)
    restraint.addPerParticleParameter("k")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    stiffness = force_constant.value_in_unit(
        unit.kilojoule_per_mole / (unit.nanometer ** 2)
    )

    topology_atoms = list(topology.atoms())
    for index in atom_indices:
        if index < 0 or index >= len(topology_atoms):
            raise IndexError(
                f"Atom index {index} is out of range for topology with "
                f"{len(topology_atoms)} atoms."
            )
        reference = reference_positions[index]
        restraint.addParticle(
            index,
            [
                stiffness,
                reference[0].value_in_unit(unit.nanometer),
                reference[1].value_in_unit(unit.nanometer),
                reference[2].value_in_unit(unit.nanometer),
            ],
        )
    return restraint


def select_restraint_atoms(topology_path: Path, selection_query: str) -> list[int]:
    """Resolve atom indices via MDAnalysis to apply positional restraints."""
    try:
        from MDAnalysis import Universe
        from MDAnalysis.core.selection import SelectionError
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing optional dependency MDAnalysis. "
            "Install it with `pip install MDAnalysis` and try again."
        ) from exc

    try:
        universe = Universe(str(topology_path), in_memory=True)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to load topology file '{topology_path}' with MDAnalysis."
        ) from exc

    try:
        atom_group = universe.select_atoms(selection_query)
    except SelectionError as exc:
        raise ValueError(
            f"Invalid MDAnalysis selection '{selection_query}': {exc}"
        ) from exc

    if atom_group.n_atoms == 0:
        raise ValueError(
            f"MDAnalysis selection '{selection_query}' matched no atoms."
        )

    return sorted(set(int(idx) for idx in atom_group.indices))
