from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from sys import stdout
from typing import Optional

import openmm
from openmm import MonteCarloBarostat, unit
from openmm.app import CheckpointReporter, DCDReporter, Simulation, StateDataReporter

from core import (
    DEFAULT_CONFIG_PATH,
    SimulationConfig,
    create_forcefield,
    create_position_restraint_force,
    ensure_output_directories,
    ensure_pdb_copy,
    load_existing_modeller,
    load_workspace_config,
    build_modeller,
    build_simulation,
    build_system,
    select_restraint_atoms,
    write_minimized_structure,
    write_topology,
)


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
    ensure_output_directories(config, checkpoint_path)
    ensure_pdb_copy(config)

    forcefield = create_forcefield(config.force_field_files)
    if restart:
        if not config.topology_path.exists():
            raise FileNotFoundError(
                f"Topology file '{config.topology_path}' not found. "
                "Cannot restart without saved topology."
            )
        modeller = load_existing_modeller(config.topology_path)
    else:
        modeller = build_modeller(forcefield, config)
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
    workspace_config = load_workspace_config(args.config)
    simulation_config = workspace_config.simulation
    checkpoint_path = args.checkpoint or simulation_config.checkpoint_path
    main(simulation_config, args.restart, checkpoint_path, args.until)
