from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import shutil
import json
import os
import random
from math import ceil
from pathlib import Path
from sys import stdout
from typing import Optional, Sequence

from datetime import datetime, timezone

import numpy as np
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
class ReplicaConfig:
    count: int
    directory_pattern: str
    seed_start: int
    seed_stride: int
    metadata_filename: str


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
    replicas: ReplicaConfig


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
    minimized_path = Path(paths["minimized"])
    trajectory_path = Path(paths["trajectory"])
    log_path = Path(paths["log"])
    checkpoint_path = Path(paths.get("checkpoint", "checkpoint.chk"))

    replicas_raw = raw.get("replicas", {})
    try:
        replica_count = int(replicas_raw.get("count", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("replicas.count must be an integer.") from exc
    if replica_count <= 0:
        raise ValueError("replicas.count must be a positive integer.")
    directory_pattern = replicas_raw.get(
        "directory_pattern", "replica_{replica_id:03d}"
    )
    try:
        seed_start = int(replicas_raw.get("seed_start", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("replicas.seed_start must be an integer.") from exc
    try:
        seed_stride = int(replicas_raw.get("seed_stride", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("replicas.seed_stride must be an integer.") from exc
    if seed_stride <= 0:
        raise ValueError("replicas.seed_stride must be positive.")
    metadata_filename = replicas_raw.get("metadata_filename", "metadata.json")
    replicas_cfg = ReplicaConfig(
        count=replica_count,
        directory_pattern=directory_pattern,
        seed_start=seed_start,
        seed_stride=seed_stride,
        metadata_filename=metadata_filename,
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
        replicas=replicas_cfg,
    )


def determine_replica_id(
    config: SimulationConfig, explicit: Optional[int]
) -> int:
    if explicit is not None:
        candidate = explicit
    else:
        candidate = None
        for env_var in ("REPLICA_ID", "PBS_ARRAY_INDEX", "SLURM_ARRAY_TASK_ID"):
            value = os.getenv(env_var)
            if value is None:
                continue
            try:
                candidate = int(value)
            except ValueError:
                continue
            else:
                break
        if candidate is None:
            candidate = 0
    if candidate < 0 or candidate >= config.replicas.count:
        raise ValueError(
            f"Replica id {candidate} is out of range for "
            f"{config.replicas.count} configured replicas."
        )
    return candidate


def determine_seed(
    config: SimulationConfig, replica_id: int, explicit: Optional[int]
) -> int:
    if explicit is not None:
        return explicit
    return config.replicas.seed_start + replica_id * config.replicas.seed_stride


def resolve_replica_directory(config: SimulationConfig, replica_id: int) -> Path:
    pattern = config.replicas.directory_pattern
    if not pattern:
        return config.simulation_dir
    try:
        subdir = pattern.format(replica_id=replica_id, replica=replica_id)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Failed to format replica directory pattern '{pattern}' "
            f"with replica_id={replica_id}."
        ) from exc
    return config.simulation_dir / subdir


def resolve_run_environment(
    config: SimulationConfig,
    checkpoint_override: Optional[Path],
    replica_id: int,
    metadata_override: Optional[Path] = None,
) -> tuple[SimulationConfig, Path, Path]:
    replica_dir = resolve_replica_directory(config, replica_id)

    def resolve_output(path: Path) -> Path:
        if path.is_absolute():
            return path
        return replica_dir / path

    checkpoint_source = checkpoint_override or config.checkpoint_path
    checkpoint_path = (
        checkpoint_source
        if checkpoint_source.is_absolute()
        else resolve_output(checkpoint_source)
    )

    resolved_config = replace(
        config,
        minimized_path=resolve_output(config.minimized_path),
        trajectory_path=resolve_output(config.trajectory_path),
        log_path=resolve_output(config.log_path),
        checkpoint_path=checkpoint_path,
    )

    metadata_path = metadata_override or Path(config.replicas.metadata_filename)
    if not metadata_path.is_absolute():
        metadata_path = replica_dir / metadata_path

    return resolved_config, replica_dir, metadata_path


def write_run_metadata(
    path: Path,
    config: SimulationConfig,
    replica_id: int,
    seed: int,
    stage: str,
    until_ns: Optional[float],
    target_production_steps: int,
    executed: dict[str, int],
    checkpoint_used: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "replica_id": replica_id,
        "seed": seed,
        "stage": stage,
        "run_id": config.run_id,
        "paths": {
            "trajectory": str(config.trajectory_path),
            "log": str(config.log_path),
            "checkpoint": str(checkpoint_used),
            "minimized": str(config.minimized_path),
            "topology": str(config.topology_path),
        },
        "thermodynamics": {
            "temperature_K": config.temperature.value_in_unit(unit.kelvin),
            "pressure_bar": config.pressure.value_in_unit(unit.bar),
            "step_size_ps": config.step_size.value_in_unit(unit.picoseconds),
        },
        "reporting": {
            "dcd_interval": config.dcd_interval,
            "stdout_interval": config.stdout_interval,
            "log_interval": config.log_interval,
        },
        "replica": {
            "directory_pattern": config.replicas.directory_pattern,
            "metadata_file": path.name,
        },
        "targets": {
            "until_ns": until_ns,
            "production_steps": target_production_steps,
        },
        "executed_steps": executed,
    }
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


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
    modeller: Modeller,
    system: openmm.System,
    config: SimulationConfig,
    seed: Optional[int] = None,
) -> Simulation:
    integrator = LangevinMiddleIntegrator(
        config.temperature, config.friction_coefficient, config.step_size
    )
    if seed is not None:
        try:
            integrator.setRandomNumberSeed(int(seed))
        except AttributeError:
            pass
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
    seed: Optional[int] = None,
) -> None:
    barostat = MonteCarloBarostat(pressure, temperature)
    if seed is not None:
        barostat.setRandomNumberSeed(int(seed))
    system.addForce(barostat)
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
    base_config: SimulationConfig,
    restart: bool,
    checkpoint_override: Optional[Path],
    until_ns: Optional[float],
    replica_id_arg: Optional[int],
    seed_arg: Optional[int],
    stage: str,
) -> None:
    stage_normalized = stage.lower()
    if stage_normalized not in {"full", "production"}:
        raise ValueError("--stage must be 'full' or 'production'.")
    if stage_normalized == "production" and not restart:
        raise ValueError(
            "--stage production requires --restart so the simulation can continue "
            "from an existing checkpoint."
        )

    replica_id = determine_replica_id(base_config, replica_id_arg)
    seed = determine_seed(base_config, replica_id, seed_arg)
    random.seed(seed)
    np.random.seed(seed)

    config, replica_dir, metadata_path = resolve_run_environment(
        base_config, checkpoint_override, replica_id
    )
    checkpoint_path = config.checkpoint_path

    required_dirs = {
        config.run_root,
        config.run_dir,
        config.initial_dir,
        config.simulation_dir,
        replica_dir,
        config.analysis_dir,
        config.topology_path.parent,
        config.minimized_path.parent,
        config.trajectory_path.parent,
        config.log_path.parent,
        checkpoint_path.parent,
        metadata_path.parent,
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
        if not config.topology_path.exists():
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

    simulation = build_simulation(modeller, system, config, seed=seed)
    attach_reporters(simulation, config, checkpoint_path, restart)
    target_production_steps = compute_production_steps(config, until_ns)

    executed_steps = {
        "minimization": 0,
        "nvt": 0,
        "npt": 0,
        "production": 0,
    }

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
        else:
            print(f"Continuing production for {remaining_steps} steps")
            run_production(simulation, remaining_steps)
            executed_steps["production"] = remaining_steps
        write_run_metadata(
            metadata_path,
            config,
            replica_id,
            seed,
            stage_normalized,
            until_ns,
            target_production_steps,
            executed_steps,
            checkpoint_path,
        )
        return

    simulation.context.setVelocitiesToTemperature(config.temperature, seed)

    print("Minimizing energy")
    simulation.minimizeEnergy()
    executed_steps["minimization"] = 1
    write_minimized_structure(simulation, config.minimized_path)

    if config.nvt_steps > 0:
        run_nvt(simulation, config.nvt_steps)
        executed_steps["nvt"] = config.nvt_steps

    if config.npt_steps > 0:
        run_npt(
            simulation,
            system,
            config.npt_steps,
            config.pressure,
            config.temperature,
            seed=seed,
        )
        executed_steps["npt"] = config.npt_steps

    if restraint_index is not None:
        system.removeForce(restraint_index)
        simulation.context.reinitialize(preserveState=True)
        simulation.context.setVelocitiesToTemperature(config.temperature, seed)

    print(f"Production target: {target_production_steps} steps")
    if target_production_steps > 0:
        run_production(simulation, target_production_steps)
        executed_steps["production"] = target_production_steps

    write_run_metadata(
        metadata_path,
        config,
        replica_id,
        seed,
        stage_normalized,
        until_ns,
        target_production_steps,
        executed_steps,
        checkpoint_path,
    )


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
    parser.add_argument(
        "--replica-id",
        type=int,
        help="Replica index (0-based). Defaults to REPLICA_ID/PBS_ARRAY_INDEX or 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed override for dynamics and stochastic components.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        help="Run stage: 'full' (default) runs minimisation/equilibration/production; "
        "'production' continues from checkpoint only.",
    )
    args = parser.parse_args()
    simulation_config = load_config(args.config)
    main(
        simulation_config,
        args.restart,
        args.checkpoint,
        args.until,
        args.replica_id,
        args.seed,
        args.stage,
    )
