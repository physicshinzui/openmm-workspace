from __future__ import annotations

import shutil
from math import ceil
from pathlib import Path
from sys import stdout
from typing import Optional, Sequence

import openmm
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, unit
from openmm.app import (
    CheckpointReporter,
    DCDReporter,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    PME,
    Simulation,
    StateDataReporter,
    Topology,
)

from md_config import SimulationConfig


def create_position_restraint_force(
    topology: Topology,
    reference_positions: unit.Quantity,
    force_constant_kj_per_mol_nm2: float,
    atom_indices: Sequence[int],
) -> openmm.CustomExternalForce:
    """Build a harmonic position restraint force for a subset of atoms."""

    energy = "0.5 * k * periodicdistance(x, y, z, x0, y0, z0)^2"
    restraint = openmm.CustomExternalForce(energy)
    restraint.addPerParticleParameter("k")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

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
                force_constant_kj_per_mol_nm2,
                reference[0].value_in_unit(unit.nanometer),
                reference[1].value_in_unit(unit.nanometer),
                reference[2].value_in_unit(unit.nanometer),
            ],
        )

    return restraint


def select_restraint_atoms(topology_path: Path, selection_query: str) -> list[int]:
    """Resolve atom indices for restraints using MDAnalysis selection syntax."""

    try:
        from MDAnalysis import Universe
        from MDAnalysis.core.selection import SelectionError
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(
            "Missing optional dependency MDAnalysis. "
            "Install it with `pip install MDAnalysis` and try again."
        ) from exc

    try:
        universe = Universe(str(topology_path), in_memory=True)
    except Exception as exc:  # pragma: no cover - propagate loader issues
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


def build_modeller(forcefield: ForceField, config: SimulationConfig) -> Modeller:
    pdb = PDBFile(str(config.paths.pdb_path))
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield, pH=7.0)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        boxShape="cube",
        padding=config.system.solvent_padding_nm * unit.nanometer,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=config.system.ionic_strength_molar * unit.molar,
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
        nonbondedCutoff=config.system.nonbonded_cutoff_nm * unit.nanometer,
        constraints=HBonds,
        removeCMMotion=True,
    )
    if config.system.hydrogen_mass_amu is not None:
        kwargs["hydrogenMass"] = config.system.hydrogen_mass_amu * unit.amu
    return forcefield.createSystem(modeller.topology, **kwargs)


def build_simulation(
    modeller: Modeller, system: openmm.System, config: SimulationConfig
) -> Simulation:
    integrator = LangevinMiddleIntegrator(
        config.thermodynamics.temperature_kelvin * unit.kelvin,
        config.thermodynamics.friction_coefficient_per_ps / unit.picosecond,
        config.thermodynamics.step_size_ps * unit.picoseconds,
    )
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    return simulation


def write_topology(modeller: Modeller, path: Path) -> None:
    with path.open("w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)


def write_minimized_structure(simulation: Simulation, path: Path) -> None:
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    with path.open("w") as handle:
        PDBFile.writeFile(simulation.topology, positions, handle)


def attach_reporters(
    simulation: Simulation,
    config: SimulationConfig,
    checkpoint_path: Path,
    restart: bool,
) -> None:
    simulation.reporters.extend(
        [
            DCDReporter(
                str(config.paths.trajectory_path),
                config.reporting.dcd_interval,
                enforcePeriodicBox=True,
                append=restart,
            ),
            StateDataReporter(
                stdout,
                config.reporting.stdout_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
            ),
            StateDataReporter(
                str(config.paths.log_path),
                config.reporting.log_interval,
                separator=",",
                append=restart,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
            ),
            CheckpointReporter(
                str(checkpoint_path),
                config.reporting.log_interval,
            ),
        ]
    )


def run_nvt(simulation: Simulation, steps: int) -> None:
    print("Running NVT")
    simulation.step(steps)


def run_npt(simulation: Simulation, system: openmm.System, config: SimulationConfig) -> None:
    system.addForce(
        MonteCarloBarostat(
            config.thermodynamics.pressure_bar * unit.bar,
            config.thermodynamics.temperature_kelvin * unit.kelvin,
        )
    )
    simulation.context.reinitialize(preserveState=True)
    print("Running NPT")
    simulation.step(config.simulation.npt_steps)


def run_production(simulation: Simulation, steps: int) -> None:
    print("Running production")
    simulation.step(steps)


def compute_target_production_steps(
    config: SimulationConfig, until_ns: Optional[float]
) -> int:
    if until_ns is None:
        return config.simulation.production_steps
    if until_ns <= 0:
        raise ValueError("--until must be positive.")
    total_ps = until_ns * 1000.0
    steps = ceil(total_ps / config.thermodynamics.step_size_ps)
    if steps <= 0:
        raise ValueError(
            "Computed production steps must be positive. "
            "Check --until and step size settings."
        )
    return steps


def ensure_output_directories(config: SimulationConfig, checkpoint_path: Path) -> None:
    required_dirs = {
        config.paths.output_root,
        config.paths.run_root,
        config.paths.initial_dir,
        config.paths.simulation_dir,
        config.paths.topology_path.parent,
        config.paths.minimized_path.parent,
        config.paths.trajectory_path.parent,
        config.paths.log_path.parent,
        checkpoint_path.parent,
    }
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def copy_input_structure(config: SimulationConfig) -> None:
    if not config.paths.pdb_path.exists():
        raise FileNotFoundError(
            f"Initial structure file '{config.paths.pdb_path}' not found."
        )

    copy_target = config.paths.pdb_copy_path
    same_location = copy_target == config.paths.pdb_path
    if not same_location and copy_target.exists():
        try:
            same_location = config.paths.pdb_path.samefile(copy_target)
        except FileNotFoundError:
            same_location = False

    if same_location:
        return

    if (
        not copy_target.exists()
        or config.paths.pdb_path.stat().st_mtime > copy_target.stat().st_mtime
    ):
        shutil.copy2(config.paths.pdb_path, copy_target)


def prepare_modeller(
    forcefield: ForceField, config: SimulationConfig, restart: bool
) -> Modeller:
    if restart:
        if not config.paths.topology_path.exists():
            raise FileNotFoundError(
                f"Topology file '{config.paths.topology_path}' not found. "
                "Cannot restart without saved topology."
            )
        return load_existing_modeller(config.paths.topology_path)

    modeller = build_modeller(forcefield, config)
    write_topology(modeller, config.paths.topology_path)
    return modeller


def maybe_add_restraints(
    system: openmm.System, modeller: Modeller, config: SimulationConfig, restart: bool
) -> Optional[int]:
    if restart or config.restraints is None:
        return None

    restraint_indices = select_restraint_atoms(
        config.paths.topology_path, config.restraints.selection
    )
    restraint_force = create_position_restraint_force(
        modeller.topology,
        modeller.positions,
        config.restraints.force_constant_kj_per_mol_nm2,
        restraint_indices,
    )
    return system.addForce(restraint_force)


def resume_production(
    simulation: Simulation,
    config: SimulationConfig,
    checkpoint_path: Path,
    target_production_steps: int,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

    print(f"Loading checkpoint from {checkpoint_path}")
    simulation.loadCheckpoint(str(checkpoint_path))
    current_step = simulation.currentStep
    print(f"Restarted at step {current_step}")

    pre_production_steps = (
        config.simulation.nvt_steps + config.simulation.npt_steps
    )
    completed_production_steps = max(current_step - pre_production_steps, 0)
    remaining_steps = target_production_steps - completed_production_steps
    if remaining_steps <= 0:
        print(
            "Target production time already reached. "
            "No additional steps will be run."
        )
        return

    print(f"Continuing production for {remaining_steps} steps")
    run_production(simulation, remaining_steps)


def run_fresh_simulation(
    simulation: Simulation,
    system: openmm.System,
    config: SimulationConfig,
    restraint_index: Optional[int],
    target_production_steps: int,
) -> None:
    print("Minimizing energy")
    simulation.minimizeEnergy()
    write_minimized_structure(simulation, config.paths.minimized_path)
    run_nvt(simulation, config.simulation.nvt_steps)
    run_npt(simulation, system, config)

    if restraint_index is not None:
        system.removeForce(restraint_index)
        simulation.context.reinitialize(preserveState=True)

    print(f"Production target: {target_production_steps} steps")
    run_production(simulation, target_production_steps)


def run_simulation(
    config: SimulationConfig,
    restart: bool = False,
    checkpoint_path: Optional[Path] = None,
    until_ns: Optional[float] = None,
) -> None:
    effective_checkpoint = checkpoint_path or config.paths.checkpoint_path
    ensure_output_directories(config, effective_checkpoint)
    copy_input_structure(config)

    forcefield = ForceField(*config.force_field_files)
    modeller = prepare_modeller(forcefield, config, restart)
    system = build_system(modeller, forcefield, config)
    restraint_index = maybe_add_restraints(system, modeller, config, restart)

    simulation = build_simulation(modeller, system, config)
    attach_reporters(simulation, config, effective_checkpoint, restart)
    target_production_steps = compute_target_production_steps(config, until_ns)

    if restart:
        resume_production(
            simulation,
            config,
            effective_checkpoint,
            target_production_steps,
        )
        return

    run_fresh_simulation(
        simulation,
        system,
        config,
        restraint_index,
        target_production_steps,
    )
