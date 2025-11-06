from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from openmm import unit

from core import (
    DEFAULT_CONFIG_PATH,
    HREXConfig,
    SimulationConfig,
    SystemBundle,
    build_modeller,
    build_simulation,
    build_system,
    create_forcefield,
    create_position_restraint_force,
    ensure_output_directories,
    ensure_pdb_copy,
    load_existing_modeller,
    load_workspace_config,
    select_restraint_atoms,
    write_topology,
)
from alchemy import AlchemicalPreparation, prepare_alchemical_system
from hrex import (
    HREXRunConfig,
    ReplicaExchangeController,
    build_compound_states,
    build_replica_exchange_sampler,
    build_sampler_state,
    create_reporter,
    resume_from_storage,
    dump_storage_to_text,
    write_iteration_log,
    write_dcd_trajectories,
)


def _ensure_hrex_directories(
    sim_config: SimulationConfig, hrex_config: HREXConfig
) -> None:
    extra_paths: list[Path] = [
        hrex_config.paths.storage_path,
        hrex_config.paths.analysis_dir / "placeholder",
        hrex_config.paths.trajectory_dir / "placeholder",
    ]
    ensure_output_directories(sim_config, extra_paths=extra_paths)
    for directory in {
        hrex_config.paths.root,
        hrex_config.paths.storage_path.parent,
        hrex_config.paths.analysis_dir,
        hrex_config.paths.trajectory_dir,
    }:
        directory.mkdir(parents=True, exist_ok=True)


def _resolve_hrex_config(workspace_config, name: Optional[str]) -> HREXConfig:
    if not workspace_config.hrex:
        raise SystemExit("No HREX configuration defined in config file.")
    if name is None:
        if len(workspace_config.hrex) == 1:
            return next(iter(workspace_config.hrex.values()))
        available = ", ".join(sorted(workspace_config.hrex))
        raise SystemExit(f"Multiple HREX configurations available: {available}. Use --hrex to select one.")
    try:
        return workspace_config.hrex[name]
    except KeyError as exc:
        available = ", ".join(sorted(workspace_config.hrex))
        raise SystemExit(
            f"Unknown HREX configuration '{name}'. Available entries: {available}."
        ) from exc


def _prepare_modeller(
    sim_config: SimulationConfig, forcefield
) -> SystemBundle:
    if sim_config.topology_path.exists():
        modeller = load_existing_modeller(sim_config.topology_path)
    else:
        modeller = build_modeller(forcefield, sim_config)
        write_topology(modeller, sim_config)
    system = build_system(modeller, forcefield, sim_config)
    bundle = SystemBundle(forcefield=forcefield, modeller=modeller, system=system)
    return bundle


def _apply_position_restraints(
    bundle: SystemBundle, sim_config: SimulationConfig, hrex_config: HREXConfig
) -> None:
    restraint_cfg = hrex_config.position_restraints or sim_config.position_restraints
    if restraint_cfg is None:
        return
    atom_indices = select_restraint_atoms(
        sim_config.topology_path, restraint_cfg.selection
    )
    restraint = create_position_restraint_force(
        bundle.topology,
        bundle.positions,
        restraint_cfg.force_constant,
        atom_indices,
    )
    bundle.system.addForce(restraint)


def _minimize(bundle: SystemBundle, sim_config: SimulationConfig, skip_minimize: bool):
    simulation = build_simulation(bundle.modeller, bundle.system, sim_config)
    if skip_minimize:
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        return state.getPositions(), simulation.topology.getPeriodicBoxVectors()
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = state.getPositions()
    box_vectors = simulation.topology.getPeriodicBoxVectors()
    bundle.modeller.positions = positions
    return positions, box_vectors


def _prepare_alchemical(
    bundle: SystemBundle, sim_config: SimulationConfig, hrex_config: HREXConfig
) -> AlchemicalPreparation:
    return prepare_alchemical_system(
        base_system=bundle.system,
        topology_path=sim_config.topology_path,
        region_cfg=hrex_config.alchemical_region,
    )


def _build_run_config(
    hrex_config: HREXConfig,
    args: argparse.Namespace,
) -> HREXRunConfig:
    run_cfg = HREXRunConfig.from_hrex(hrex_config)
    if args.no_minimize:
        run_cfg.minimize = False
    if args.equil_iterations is not None:
        run_cfg.equilibration_iterations = args.equil_iterations
    if args.production_iterations is not None:
        run_cfg.production_iterations = args.production_iterations
    return run_cfg


def run_new(
    sim_config: SimulationConfig,
    hrex_config: HREXConfig,
    *,
    skip_minimize: bool,
    overwrite: bool,
    run_cfg: HREXRunConfig,
) -> None:
    storage_path = hrex_config.paths.storage_path
    if storage_path.exists():
        if not overwrite:
            raise SystemExit(
                f"Storage file '{storage_path}' already exists. Use --overwrite to replace it or --resume."
            )
        storage_path.unlink()
        checkpoint_name = hrex_config.paths.checkpoint_filename
        if checkpoint_name is not None:
            checkpoint_path = hrex_config.paths.root / checkpoint_name
        else:
            checkpoint_path = storage_path.with_name(
                f"{storage_path.stem}_checkpoint{storage_path.suffix}"
            )
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    forcefield = create_forcefield(sim_config.force_field_files)
    ensure_pdb_copy(sim_config)
    bundle = _prepare_modeller(sim_config, forcefield)
    _apply_position_restraints(bundle, sim_config, hrex_config)
    positions, box_vectors = _minimize(bundle, sim_config, skip_minimize)
    alchemical_prep = _prepare_alchemical(bundle, sim_config, hrex_config)
    thermodynamic_states = build_compound_states(
        alchemical_prep, sim_config, hrex_config
    )
    sampler_state = build_sampler_state(
        positions=positions,
        box_vectors=box_vectors,
    )
    sampler = build_replica_exchange_sampler(hrex_config)
    reporter = create_reporter(hrex_config, mode=None)
    controller = ReplicaExchangeController(sampler, reporter, hrex_config)
    try:
        controller.create(
            thermodynamic_states=thermodynamic_states,
            sampler_state=sampler_state,
        )
        controller.run(run_cfg)
    finally:
        controller.close()

    if hrex_config.reporting.position_interval > 0:
        frame_time_ps = (
            hrex_config.mcmc.n_steps_per_iteration
            * sim_config.step_size.value_in_unit(unit.picoseconds)
            * hrex_config.reporting.position_interval
        )
        write_dcd_trajectories(
            storage_path=hrex_config.paths.storage_path,
            topology_path=sim_config.topology_path,
            output_dir=hrex_config.paths.trajectory_dir,
            frame_time_ps=frame_time_ps,
        )
    log_path = hrex_config.paths.analysis_dir / "hrex_log.csv"
    write_iteration_log(
        storage_path=hrex_config.paths.storage_path,
        output_path=log_path,
    )
    text_report_path = hrex_config.paths.analysis_dir / "hrex_summary.txt"
    dump_storage_to_text(
        storage_path=hrex_config.paths.storage_path,
        output_path=text_report_path,
    )


def run_resume(
    hrex_config: HREXConfig,
    production_iterations: Optional[int],
) -> None:
    storage_path = hrex_config.paths.storage_path
    if not storage_path.exists():
        raise SystemExit(
            f"Cannot resume; storage file '{storage_path}' does not exist."
        )
    sampler, reporter = resume_from_storage(storage_path)
    controller = ReplicaExchangeController(sampler, reporter, hrex_config)
    try:
        controller.resume(production_iterations)
    finally:
        controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Hamiltonian replica exchange using openmmtools."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to workspace configuration YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--hrex",
        type=str,
        help="Name of the HREX configuration block to use.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing HREX run using the configured storage file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HREX storage file when starting a new run.",
    )
    parser.add_argument(
        "--no-minimize",
        action="store_true",
        help="Skip energy minimisation before launching HREX.",
    )
    parser.add_argument(
        "--equil-iterations",
        type=int,
        help="Override number of equilibration iterations.",
    )
    parser.add_argument(
        "--production-iterations",
        type=int,
        help="Override number of production iterations.",
    )
    args = parser.parse_args()

    workspace_config = load_workspace_config(args.config)
    sim_config = workspace_config.simulation
    hrex_config = _resolve_hrex_config(workspace_config, args.hrex)
    _ensure_hrex_directories(sim_config, hrex_config)

    if args.resume:
        run_resume(hrex_config, args.production_iterations)
        print(f"Extended HREX run recorded in {hrex_config.paths.storage_path}")
        return

    run_cfg = _build_run_config(hrex_config, args)
    run_new(
        sim_config,
        hrex_config,
        skip_minimize=args.no_minimize,
        overwrite=args.overwrite,
        run_cfg=run_cfg,
    )
    print(f"HREX run completed. Storage written to {hrex_config.paths.storage_path}")


if __name__ == "__main__":
    main()
