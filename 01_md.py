from __future__ import annotations

import argparse
from dataclasses import dataclass
import shutil
from math import ceil
from pathlib import Path
from sys import stdout
from typing import Optional, Sequence

import openmm
from openmm import MonteCarloBarostat, LangevinMiddleIntegrator, unit

from openmm.app import (
    DCDReporter,
    ForceField,
    HBonds,
    Modeller,
    CheckpointReporter,
    PME,
    PDBFile,
    Simulation,
    StateDataReporter,
    Topology,
)

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
    raise SystemExit(
        "Missing optional dependency PyYAML. "
        "Install it with `pip install pyyaml` and try again."
    ) from exc


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass(frozen=True)
class RestraintConfig:
    force_constant: unit.Quantity
    selection: str


@dataclass(frozen=True)
class SimulationConfig:
    pdb_path: Path
    pdb_copy_path: Path
    run_root: Path
    run_dir: Path
    initial_dir: Path
    simulation_dir: Path
    analysis_dir: Path
    run_id: str
    topology_path: Path
    minimized_path: Path
    trajectory_path: Path
    log_path: Path
    checkpoint_path: Path
    force_field_files: tuple[str, ...]
    temperature: unit.Quantity
    pressure: unit.Quantity
    friction_coefficient: unit.Quantity
    step_size: unit.Quantity
    nonbonded_cutoff: unit.Quantity
    solvent_padding: unit.Quantity
    ionic_strength: unit.Quantity
    hydrogen_mass: Optional[unit.Quantity]
    nvt_steps: int
    npt_steps: int
    production_steps: int
    dcd_interval: int
    stdout_interval: int
    log_interval: int
    position_restraints: Optional[RestraintConfig]


def load_config(path: Path) -> SimulationConfig:
    with path.open() as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(f"Configuration file {path} is empty.")

    try:
        paths = raw["paths"]
        thermodynamics = raw["thermodynamics"]
        system_cfg = raw["system"]
        simulation_cfg = raw["simulation"]
        reporting = raw["reporting"]
        force_fields: Sequence[str] = raw["force_fields"]
    except KeyError as exc:
        raise KeyError(
            "Missing top-level configuration section "
            f"{exc} in YAML file {path}."
        ) from exc

    restraints_raw = raw.get("restraints")
    restraints_cfg: Optional[RestraintConfig] = None
    if restraints_raw:
        try:
            force_constant = restraints_raw["force_constant"]
            selection = restraints_raw["selection"]
        except KeyError as exc:
            raise KeyError(
                "Restraints configuration requires 'force_constant' and 'selection' keys."
            ) from exc
        restraints_cfg = RestraintConfig(
            force_constant=force_constant
            * unit.kilojoule_per_mole
            / (unit.nanometer ** 2),
            selection=selection,
        )

    pdb_path = Path(paths["pdb"])
    run_root = Path(paths.get("output_root", "data/md_runs"))
    run_id = paths.get("run_id", "default")
    run_dir = run_root / pdb_path.stem
    initial_dir = run_dir / "initial"
    simulation_dir = run_dir / "simulations" / run_id
    analysis_dir = run_dir / "analysis"
    pdb_copy_path = initial_dir / pdb_path.name

    topology_path_raw = Path(paths["topology"])
    topology_path = (
        topology_path_raw
        if topology_path_raw.is_absolute()
        else initial_dir / topology_path_raw
    )
    minimized_path_raw = Path(paths["minimized"])
    minimized_path = (
        minimized_path_raw
        if minimized_path_raw.is_absolute()
        else simulation_dir / minimized_path_raw
    )
    trajectory_path_raw = Path(paths["trajectory"])
    trajectory_path = (
        trajectory_path_raw
        if trajectory_path_raw.is_absolute()
        else simulation_dir / trajectory_path_raw
    )
    log_path_raw = Path(paths["log"])
    log_path = (
        log_path_raw
        if log_path_raw.is_absolute()
        else simulation_dir / log_path_raw
    )
    checkpoint_path_raw = Path(paths.get("checkpoint", "checkpoint.chk"))
    checkpoint_path = (
        checkpoint_path_raw
        if checkpoint_path_raw.is_absolute()
        else simulation_dir / checkpoint_path_raw
    )

    return SimulationConfig(
        pdb_path=pdb_path,
        pdb_copy_path=pdb_copy_path,
        run_root=run_root,
        run_dir=run_dir,
        initial_dir=initial_dir,
        simulation_dir=simulation_dir,
        analysis_dir=analysis_dir,
        run_id=run_id,
        topology_path=topology_path,
        minimized_path=minimized_path,
        trajectory_path=trajectory_path,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        force_field_files=tuple(force_fields),
        temperature=thermodynamics["temperature"] * unit.kelvin,
        pressure=thermodynamics["pressure"] * unit.bar,
        friction_coefficient=thermodynamics["friction_coefficient"]
        / unit.picosecond,
        step_size=thermodynamics["step_size"] * unit.picoseconds,
        nonbonded_cutoff=system_cfg["nonbonded_cutoff"] * unit.nanometer,
        solvent_padding=system_cfg["solvent_padding"] * unit.nanometer,
        ionic_strength=system_cfg["ionic_strength"] * unit.molar,
        hydrogen_mass=(
            system_cfg.get("hydrogen_mass") * unit.amu
            if "hydrogen_mass" in system_cfg
            else None
        ),
        nvt_steps=simulation_cfg["nvt_steps"],
        npt_steps=simulation_cfg["npt_steps"],
        production_steps=simulation_cfg["production_steps"],
        dcd_interval=reporting["dcd_interval"],
        stdout_interval=reporting["stdout_interval"],
        log_interval=reporting["log_interval"],
        position_restraints=restraints_cfg,
    )


def create_position_restraint_force(
    topology: Topology,
    reference_positions: unit.Quantity,
    force_constant: unit.Quantity,
    atom_indices: Sequence[int],
) -> openmm.CustomExternalForce:
    """Build a harmonic position restraint force for a subset of atoms."""

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


def build_simulation(
    modeller: Modeller, system: openmm.System, config: SimulationConfig
) -> Simulation:
    integrator = LangevinMiddleIntegrator(
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


def attach_reporters(
    simulation: Simulation,
    config: SimulationConfig,
    checkpoint_path: Path,
    restart: bool,
) -> None:
    simulation.reporters.extend(
        [
            DCDReporter(
                str(config.trajectory_path),
                config.dcd_interval,
                enforcePeriodicBox=True,
                append=restart,
            ),
            StateDataReporter(
                stdout,
                config.stdout_interval,
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
                str(config.log_path),
                config.log_interval,
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
                config.log_interval,
            ),
        ]
    )


def run_nvt(simulation: Simulation, steps: int) -> None:
    print("Running NVT")
    simulation.step(steps)


def run_npt(
    simulation: Simulation,
    system: openmm.System,
    steps: int,
    pressure: unit.Quantity,
    temperature: unit.Quantity,
) -> None:
    system.addForce(MonteCarloBarostat(pressure, temperature))
    simulation.context.reinitialize(preserveState=True)
    print("Running NPT")
    simulation.step(steps)


def run_production(simulation: Simulation, steps: int) -> None:
    print("Running production")
    simulation.step(steps)


def compute_production_steps(
    config: SimulationConfig, until_ns: Optional[float]
) -> int:
    if until_ns is None:
        return config.production_steps
    if until_ns <= 0:
        raise ValueError("--until must be positive.")
    step_size_ps = config.step_size.value_in_unit(unit.picoseconds)
    total_ps = until_ns * 1000.0
    steps = ceil(total_ps / step_size_ps)
    if steps <= 0:
        raise ValueError(
            "Computed production steps must be positive. "
            "Check --until and step size settings."
        )
    return steps


def main(
    config: SimulationConfig,
    restart: bool,
    checkpoint_path: Path,
    until_ns: Optional[float],
) -> None:
    required_dirs = {
        config.run_root,
        config.run_dir,
        config.initial_dir,
        config.simulation_dir,
        config.analysis_dir,
        config.topology_path.parent,
        config.minimized_path.parent,
        config.trajectory_path.parent,
        config.log_path.parent,
        checkpoint_path.parent,
    }
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

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
    if not same_location:
        if (
            not copy_target.exists()
            or config.pdb_path.stat().st_mtime > copy_target.stat().st_mtime
        ):
            shutil.copy2(config.pdb_path, copy_target)

    forcefield = ForceField(*config.force_field_files)
    if restart:
        if not config.topology_path.exists():
            raise FileNotFoundError(
                f"Topology file '{config.topology_path}' not found. "
                "Cannot restart without saved topology."
            )
        modeller = load_existing_modeller(config.topology_path)
    else:
        modeller = build_modeller(forcefield, config)
    if not restart:
        write_topology(modeller, config)

    system = build_system(modeller, forcefield, config)
    restraint_index: Optional[int] = None
    if not restart and config.position_restraints is not None:
        restraint_indices = select_restraint_atoms(
            config.topology_path, config.position_restraints.selection
        )
        restraint_force = create_position_restraint_force(
            modeller.topology,
            modeller.positions,
            config.position_restraints.force_constant,
            restraint_indices,
        )
        restraint_index = system.addForce(restraint_force)

    simulation = build_simulation(modeller, system, config)
    attach_reporters(simulation, config, checkpoint_path, restart)
    target_production_steps = compute_production_steps(config, until_ns)

    if restart:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file '{checkpoint_path}' not found."
            )
        print(f"Loading checkpoint from {checkpoint_path}")
        simulation.loadCheckpoint(str(checkpoint_path))
        current_step = simulation.currentStep
        print(f"Restarted at step {current_step}")
        pre_production_steps = config.nvt_steps + config.npt_steps
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
        return

    print("Minimizing energy")
    simulation.minimizeEnergy()
    write_minimized_structure(simulation, config.minimized_path)
    run_nvt(simulation, config.nvt_steps)
    run_npt(
        simulation,
        system,
        config.npt_steps,
        config.pressure,
        config.temperature,
    )
    if restraint_index is not None:
        system.removeForce(restraint_index)
        simulation.context.reinitialize(preserveState=True)
    print(f"Production target: {target_production_steps} steps")
    run_production(simulation, target_production_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenMM MD simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override checkpoint path (defaults to value from config).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Resume from checkpoint and run production stage only.",
    )
    parser.add_argument(
        "--until",
        type=float,
        help="Target production time in nanoseconds. Overrides production_steps.",
    )
    args = parser.parse_args()
    simulation_config = load_config(args.config)
    checkpoint_path = args.checkpoint or simulation_config.checkpoint_path
    main(simulation_config, args.restart, checkpoint_path, args.until)
