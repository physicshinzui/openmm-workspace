from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

import openmm
from openmm.app import (
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    PME,
    Simulation,
)

from .config import SimulationConfig


@dataclass(frozen=True)
class SystemBundle:
    forcefield: ForceField
    modeller: Modeller
    system: openmm.System

    @property
    def topology(self):
        return self.modeller.topology

    @property
    def positions(self):
        return self.modeller.positions

    @property
    def box_vectors(self):
        vectors = self.modeller.topology.getPeriodicBoxVectors()
        if vectors is not None:
            return vectors
        try:
            return self.system.getDefaultPeriodicBoxVectors()
        except AttributeError:
            return None


def create_forcefield(force_field_files: Sequence[str]) -> ForceField:
    if not force_field_files:
        raise ValueError("At least one force field XML file must be provided.")
    return ForceField(*force_field_files)


def build_modeller(forcefield: ForceField, config: SimulationConfig) -> Modeller:
    pdb = PDBFile(str(config.pdb_path))
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield, pH=7.0)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        boxShape="cube",
        padding=config.solvent_padding,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=config.ionic_strength,
    )
    return modeller


def load_existing_modeller(topology_path: Path) -> Modeller:
    pdb = PDBFile(str(topology_path))
    return Modeller(pdb.topology, pdb.positions)


def build_system(
    modeller: Modeller, forcefield: ForceField, config: SimulationConfig
) -> openmm.System:
    kwargs = dict(
        nonbondedMethod=PME,
        nonbondedCutoff=config.nonbonded_cutoff,
        constraints=HBonds,
        removeCMMotion=True,
    )
    if config.hydrogen_mass is not None:
        kwargs["hydrogenMass"] = config.hydrogen_mass
    return forcefield.createSystem(modeller.topology, **kwargs)


def build_system_bundle(
    modeller: Modeller, forcefield: ForceField, config: SimulationConfig
) -> SystemBundle:
    system = build_system(modeller, forcefield, config)
    return SystemBundle(forcefield=forcefield, modeller=modeller, system=system)


def build_simulation(
    modeller: Modeller, system: openmm.System, config: SimulationConfig
) -> Simulation:
    integrator = openmm.LangevinMiddleIntegrator(
        config.temperature, config.friction_coefficient, config.step_size
    )
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    return simulation


def write_topology(modeller: Modeller, config: SimulationConfig) -> None:
    with config.topology_path.open("w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)


def write_minimized_structure(simulation: Simulation, path: Path) -> None:
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    with path.open("w") as handle:
        PDBFile.writeFile(simulation.topology, positions, handle)


def ensure_output_directories(
    config: SimulationConfig,
    checkpoint_path: Optional[Path] = None,
    extra_paths: Optional[Iterable[Path]] = None,
) -> None:
    directories = {
        config.run_root,
        config.run_dir,
        config.initial_dir,
        config.simulation_dir,
        config.analysis_dir,
        config.topology_path.parent,
        config.minimized_path.parent,
        config.trajectory_path.parent,
        config.log_path.parent,
    }
    if checkpoint_path is not None:
        directories.add(checkpoint_path.parent)
    if extra_paths:
        directories.update(Path(path).parent for path in extra_paths)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_pdb_copy(config: SimulationConfig) -> None:
    """Ensure the initial PDB is copied into the run directory."""
    if not config.pdb_path.exists():
        raise FileNotFoundError(
            f"Initial structure file '{config.pdb_path}' not found."
        )
    copy_target = config.pdb_copy_path
    same_location = copy_target == config.pdb_path
    if not same_location and copy_target.exists():
        try:
            same_location = config.pdb_path.samefile(copy_target)
        except FileNotFoundError:
            same_location = False
    if same_location:
        return
    if (
        not copy_target.exists()
        or config.pdb_path.stat().st_mtime > copy_target.stat().st_mtime
    ):
        shutil.copy2(config.pdb_path, copy_target)
