from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from openmm import unit

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


@dataclass(frozen=True)
class HREXPathsConfig:
    root: Path
    storage_path: Path
    checkpoint_filename: Optional[str]
    analysis_dir: Path
    trajectory_dir: Path


@dataclass(frozen=True)
class HREXProtocolConfig:
    lambda_schedule: Tuple[float, ...]
    temperature_schedule: Optional[Tuple[unit.Quantity, ...]]
    barostat_frequency: Optional[int]
    pressure: Optional[unit.Quantity]

    @property
    def n_replicas(self) -> int:
        return len(self.lambda_schedule)


@dataclass(frozen=True)
class HREXMCMCConfig:
    timestep: unit.Quantity
    collision_rate: unit.Quantity
    n_steps_per_iteration: int
    splitting: str


@dataclass(frozen=True)
class HREXIterationConfig:
    equilibration_iterations: int
    production_iterations: int
    checkpoint_interval: int


@dataclass(frozen=True)
class HREXReportingConfig:
    analysis_particle_indices: Tuple[int, ...]
    position_interval: int
    velocity_interval: int


@dataclass(frozen=True)
class HREXAlchemicalRegionConfig:
    selection: str
    annihilate_electrostatics: bool
    annihilate_sterics: bool
    softcore_alpha: Optional[float]
    softcore_beta: Optional[float]
    softcore_a: Optional[int]
    softcore_b: Optional[int]


@dataclass(frozen=True)
class HREXConfig:
    name: str
    paths: HREXPathsConfig
    protocol: HREXProtocolConfig
    mcmc: HREXMCMCConfig
    iterations: HREXIterationConfig
    reporting: HREXReportingConfig
    alchemical_region: HREXAlchemicalRegionConfig
    position_restraints: Optional[RestraintConfig]
    random_seed: Optional[int]


@dataclass(frozen=True)
class WorkspaceConfig:
    simulation: SimulationConfig
    hrex: Dict[str, HREXConfig]


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_optional_path(
    base_dir: Path, default: Path, raw_path: str | Path | None
) -> Path:
    if raw_path is None:
        return default
    return _resolve_path(base_dir, raw_path)


def _ensure_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    try:
        value = raw[key]
    except KeyError as exc:
        raise KeyError(f"Missing configuration section '{key}'.") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section '{key}' must be a mapping.")
    return value


def _load_restraints(
    restraints_raw: Mapping[str, Any] | None,
) -> Optional[RestraintConfig]:
    if not restraints_raw:
        return None
    try:
        force_constant = restraints_raw["force_constant"]
        selection = restraints_raw["selection"]
    except KeyError as exc:
        raise KeyError(
            "Restraints configuration requires 'force_constant' and 'selection'."
        ) from exc
    return RestraintConfig(
        force_constant=float(force_constant)
        * unit.kilojoule_per_mole
        / (unit.nanometer ** 2),
        selection=str(selection),
    )


def _as_float_sequence(raw: Sequence[Any], label: str) -> Tuple[float, ...]:
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a sequence of numbers.") from exc
    if not values:
        raise ValueError(f"{label} must contain at least one value.")
    return values


def _as_int_sequence(raw: Sequence[Any], label: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a sequence of integers.") from exc
    return values


def _ensure_positive_int(value: Any, label: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be an integer.") from exc
    if integer <= 0:
        raise ValueError(f"{label} must be positive.")
    return integer


def _ensure_nonnegative_int(value: Any, label: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be an integer.") from exc
    if integer < 0:
        raise ValueError(f"{label} must be non-negative.")
    return integer


def _parse_simulation_config(
    raw: Mapping[str, Any], config_dir: Path
) -> SimulationConfig:
    paths = _ensure_mapping(raw, "paths")
    thermodynamics = _ensure_mapping(raw, "thermodynamics")
    system_cfg = _ensure_mapping(raw, "system")
    simulation_cfg = _ensure_mapping(raw, "simulation")
    reporting = _ensure_mapping(raw, "reporting")

    force_fields_raw = raw.get("force_fields")
    if force_fields_raw is None:
        raise KeyError("Missing 'force_fields' list in configuration.")
    if not isinstance(force_fields_raw, Sequence):
        raise TypeError("'force_fields' must be a sequence.")
    force_fields = tuple(str(item) for item in force_fields_raw)

    pdb_path = _resolve_path(config_dir, paths["pdb"])
    run_root = _resolve_path(config_dir, paths.get("output_root", "data/md_runs"))
    run_id = str(paths.get("run_id", "default"))
    run_dir = (run_root / pdb_path.stem).resolve()
    initial_dir = (run_dir / "initial").resolve()
    simulation_dir = (run_dir / "simulations" / run_id).resolve()
    analysis_dir = (run_dir / "analysis").resolve()
    pdb_copy_path = (initial_dir / Path(pdb_path).name).resolve()

    topology_path = _resolve_optional_path(
        initial_dir, initial_dir / Path(paths["topology"]), paths.get("topology")
    )
    minimized_path = _resolve_optional_path(
        simulation_dir, simulation_dir / Path(paths["minimized"]), paths.get("minimized")
    )
    trajectory_path = _resolve_optional_path(
        simulation_dir, simulation_dir / Path(paths["trajectory"]), paths.get("trajectory")
    )
    log_path = _resolve_optional_path(
        simulation_dir, simulation_dir / Path(paths["log"]), paths.get("log")
    )
    checkpoint_path = _resolve_optional_path(
        simulation_dir,
        simulation_dir / Path(paths.get("checkpoint", "checkpoint.chk")),
        paths.get("checkpoint"),
    )

    restraints_cfg = _load_restraints(raw.get("restraints"))

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
        force_field_files=force_fields,
        temperature=float(thermodynamics["temperature"]) * unit.kelvin,
        pressure=float(thermodynamics["pressure"]) * unit.bar,
        friction_coefficient=float(thermodynamics["friction_coefficient"])
        / unit.picosecond,
        step_size=float(thermodynamics["step_size"]) * unit.picoseconds,
        nonbonded_cutoff=float(system_cfg["nonbonded_cutoff"]) * unit.nanometer,
        solvent_padding=float(system_cfg["solvent_padding"]) * unit.nanometer,
        ionic_strength=float(system_cfg["ionic_strength"]) * unit.molar,
        hydrogen_mass=(
            float(system_cfg["hydrogen_mass"]) * unit.amu
            if "hydrogen_mass" in system_cfg
            else None
        ),
        nvt_steps=_ensure_positive_int(simulation_cfg["nvt_steps"], "simulation.nvt_steps"),
        npt_steps=_ensure_positive_int(simulation_cfg["npt_steps"], "simulation.npt_steps"),
        production_steps=_ensure_positive_int(
            simulation_cfg["production_steps"], "simulation.production_steps"
        ),
        dcd_interval=_ensure_positive_int(reporting["dcd_interval"], "reporting.dcd_interval"),
        stdout_interval=_ensure_positive_int(
            reporting["stdout_interval"], "reporting.stdout_interval"
        ),
        log_interval=_ensure_positive_int(reporting["log_interval"], "reporting.log_interval"),
        position_restraints=restraints_cfg,
    )


def _parse_hrex_paths(
    name: str, raw: Mapping[str, Any], config_dir: Path, sim_config: SimulationConfig
) -> HREXPathsConfig:
    default_root = (sim_config.run_dir / "hrex" / name).resolve()
    root = (
        _resolve_path(config_dir, raw["root"])
        if "root" in raw
        else default_root
    )
    storage_default = root / "storage.nc"
    analysis_default = root / "analysis"
    trajectories_default = root / "trajectories"

    storage_path = _resolve_optional_path(root, storage_default, raw.get("storage"))
    analysis_dir = _resolve_optional_path(root, analysis_default, raw.get("analysis"))
    trajectory_dir = _resolve_optional_path(
        root, trajectories_default, raw.get("trajectories")
    )
    checkpoint_filename = raw.get("checkpoint_filename")
    if checkpoint_filename is not None:
        checkpoint_filename = str(checkpoint_filename)

    return HREXPathsConfig(
        root=root,
        storage_path=storage_path,
        checkpoint_filename=checkpoint_filename,
        analysis_dir=analysis_dir,
        trajectory_dir=trajectory_dir,
    )


def _parse_hrex_protocol(raw: Mapping[str, Any], sim_config: SimulationConfig) -> HREXProtocolConfig:
    lambda_schedule_raw = raw.get("lambda_schedule")
    if lambda_schedule_raw is None:
        raise KeyError("hrex.protocol.lambda_schedule is required.")
    lambda_schedule = _as_float_sequence(lambda_schedule_raw, "hrex.protocol.lambda_schedule")

    temperature_schedule_raw = raw.get("temperature_schedule")
    if temperature_schedule_raw is not None:
        temperature_schedule = tuple(
            float(value) * unit.kelvin for value in temperature_schedule_raw
        )
        if len(temperature_schedule) != len(lambda_schedule):
            raise ValueError(
                "hrex.protocol.temperature_schedule must match lambda_schedule length."
            )
    else:
        temperature_schedule = None

    barostat_frequency = raw.get("barostat_frequency")
    if barostat_frequency is not None:
        barostat_frequency = _ensure_positive_int(
            barostat_frequency, "hrex.protocol.barostat_frequency"
        )

    pressure_raw = raw.get("pressure")
    if pressure_raw is None:
        pressure = sim_config.pressure if barostat_frequency is not None else None
    else:
        pressure = float(pressure_raw) * unit.bar

    return HREXProtocolConfig(
        lambda_schedule=lambda_schedule,
        temperature_schedule=temperature_schedule,
        barostat_frequency=barostat_frequency,
        pressure=pressure,
    )


def _parse_hrex_mcmc(raw: Mapping[str, Any], sim_config: SimulationConfig) -> HREXMCMCConfig:
    timestep = float(raw.get("timestep", sim_config.step_size.value_in_unit(unit.picoseconds)))
    collision = float(
        raw.get(
            "collision_rate",
            sim_config.friction_coefficient.value_in_unit(unit.picosecond**-1),
        )
    )
    n_steps = _ensure_positive_int(
        raw.get("n_steps_per_iteration", 1000), "hrex.mcmc.n_steps_per_iteration"
    )
    splitting = str(raw.get("splitting", "V R O R V"))
    return HREXMCMCConfig(
        timestep=timestep * unit.picoseconds,
        collision_rate=collision / unit.picosecond,
        n_steps_per_iteration=n_steps,
        splitting=splitting,
    )


def _parse_hrex_iterations(raw: Mapping[str, Any]) -> HREXIterationConfig:
    equil = _ensure_nonnegative_int(
        raw.get("equilibration_iterations", 50), "hrex.iterations.equilibration_iterations"
    )
    prod = _ensure_positive_int(
        raw.get("production_iterations", 500), "hrex.iterations.production_iterations"
    )
    checkpoint_interval = _ensure_positive_int(
        raw.get("checkpoint_interval", 10), "hrex.iterations.checkpoint_interval"
    )
    return HREXIterationConfig(
        equilibration_iterations=equil,
        production_iterations=prod,
        checkpoint_interval=checkpoint_interval,
    )


def _parse_hrex_reporting(raw: Mapping[str, Any]) -> HREXReportingConfig:
    particle_indices_raw = raw.get("analysis_particles", [])
    if particle_indices_raw and not isinstance(particle_indices_raw, Sequence):
        raise TypeError("hrex.reporting.analysis_particles must be a sequence.")
    particle_indices = (
        _as_int_sequence(particle_indices_raw, "hrex.reporting.analysis_particles")
        if particle_indices_raw
        else tuple()
    )
    position_interval = _ensure_nonnegative_int(
        raw.get("position_interval", 0), "hrex.reporting.position_interval"
    )
    velocity_interval = _ensure_nonnegative_int(
        raw.get("velocity_interval", 0), "hrex.reporting.velocity_interval"
    )
    return HREXReportingConfig(
        analysis_particle_indices=particle_indices,
        position_interval=position_interval,
        velocity_interval=velocity_interval,
    )


def _parse_hrex_alchemical_region(raw: Mapping[str, Any]) -> HREXAlchemicalRegionConfig:
    try:
        selection = raw["selection"]
    except KeyError as exc:
        raise KeyError("hrex.alchemical_region.selection is required.") from exc
    return HREXAlchemicalRegionConfig(
        selection=str(selection),
        annihilate_electrostatics=bool(raw.get("annihilate_electrostatics", True)),
        annihilate_sterics=bool(raw.get("annihilate_sterics", True)),
        softcore_alpha=(float(raw["softcore_alpha"]) if "softcore_alpha" in raw else None),
        softcore_beta=(float(raw["softcore_beta"]) if "softcore_beta" in raw else None),
        softcore_a=(int(raw["softcore_a"]) if "softcore_a" in raw else None),
        softcore_b=(int(raw["softcore_b"]) if "softcore_b" in raw else None),
    )


def _parse_hrex_config(
    name: str,
    cfg: Mapping[str, Any],
    config_dir: Path,
    sim_config: SimulationConfig,
) -> HREXConfig:
    if not isinstance(cfg, Mapping):
        raise TypeError(f"hrex.{name} configuration must be a mapping.")

    paths_cfg = cfg.get("paths", {})
    if paths_cfg and not isinstance(paths_cfg, Mapping):
        raise TypeError(f"hrex.{name}.paths must be a mapping if provided.")
    protocol_cfg = _ensure_mapping(cfg, "protocol")
    mcmc_cfg = _ensure_mapping(cfg, "mcmc") if "mcmc" in cfg else {}
    iterations_cfg = _ensure_mapping(cfg, "iterations") if "iterations" in cfg else {}
    reporting_cfg = _ensure_mapping(cfg, "reporting") if "reporting" in cfg else {}
    region_cfg = _ensure_mapping(cfg, "alchemical_region")

    paths = _parse_hrex_paths(name, paths_cfg, config_dir, sim_config)
    protocol = _parse_hrex_protocol(protocol_cfg, sim_config)
    mcmc = _parse_hrex_mcmc(mcmc_cfg, sim_config)
    iterations = _parse_hrex_iterations(iterations_cfg)
    reporting = _parse_hrex_reporting(reporting_cfg)
    alchemical_region = _parse_hrex_alchemical_region(region_cfg)

    restraints_cfg = _load_restraints(cfg.get("restraints"))

    random_seed = cfg.get("random_seed")
    if random_seed is not None:
        random_seed = int(random_seed)

    if protocol.temperature_schedule is not None and (
        protocol.barostat_frequency is None and protocol.pressure is not None
    ):
        # Temperature-varying states still reuse the same pressure object.
        pass

    return HREXConfig(
        name=name,
        paths=paths,
        protocol=protocol,
        mcmc=mcmc,
        iterations=iterations,
        reporting=reporting,
        alchemical_region=alchemical_region,
        position_restraints=restraints_cfg,
        random_seed=random_seed,
    )


def _parse_hrex_configs(
    hrex_raw: Mapping[str, Any] | None,
    config_dir: Path,
    sim_config: SimulationConfig,
) -> Dict[str, HREXConfig]:
    if hrex_raw is None:
        return {}
    if not isinstance(hrex_raw, Mapping):
        raise TypeError("The 'hrex' section must be a mapping of named configurations.")
    configs: Dict[str, HREXConfig] = {}
    for name, cfg in hrex_raw.items():
        hrex_config = _parse_hrex_config(str(name), cfg, config_dir, sim_config)
        configs[hrex_config.name] = hrex_config
    return configs


def load_workspace_config(path: Path) -> WorkspaceConfig:
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ValueError(f"Configuration file {path} is empty.")
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"Configuration file {path} must define a mapping at the top level."
        )
    config_dir = path.parent.resolve()
    simulation_config = _parse_simulation_config(raw, config_dir)
    hrex_configs = _parse_hrex_configs(raw.get("hrex"), config_dir, simulation_config)
    return WorkspaceConfig(simulation=simulation_config, hrex=hrex_configs)
