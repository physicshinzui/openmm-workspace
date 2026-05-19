from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

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
    force_constant_kj_per_mol_nm2: float
    selection: str


@dataclass(frozen=True)
class SimulationPaths:
    config_path: Path
    input_format: str
    system_name: str
    pdb_path: Optional[Path]
    prmtop_path: Optional[Path]
    inpcrd_path: Optional[Path]
    input_copy_paths: tuple[Path, ...]
    output_root: Path
    system_root: Path
    run_root: Path
    input_dir: Path
    output_dir: Path
    run_id: str
    topology_path: Path
    minimized_path: Path
    trajectory_path: Path
    log_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class ThermodynamicsConfig:
    temperature_kelvin: float
    pressure_bar: float
    friction_coefficient_per_ps: float
    step_size_ps: float


@dataclass(frozen=True)
class SystemConfig:
    nonbonded_cutoff_nm: float
    solvent_padding_nm: Optional[float]
    ionic_strength_molar: Optional[float]
    hydrogen_mass_amu: Optional[float]


@dataclass(frozen=True)
class SimulationStages:
    nvt_steps: int
    npt_steps: int
    production_steps: int


@dataclass(frozen=True)
class ReportingConfig:
    dcd_interval: int
    stdout_interval: int
    log_interval: int


@dataclass(frozen=True)
class SimulationConfig:
    paths: SimulationPaths
    force_field_files: tuple[str, ...]
    thermodynamics: ThermodynamicsConfig
    system: SystemConfig
    simulation: SimulationStages
    reporting: ReportingConfig
    restraints: Optional[RestraintConfig]


def load_yaml_document(path: Path) -> Any:
    try:
        with path.open() as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"YAML file '{path}' was not found.") from exc


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = load_yaml_document(path)
    if data is None:
        raise ValueError(f"YAML file '{path}' is empty.")
    if not isinstance(data, dict):
        raise ValueError(f"YAML file '{path}' must contain a mapping.")
    return data


def load_yaml_list(path: Path) -> list[dict[str, Any]]:
    data = load_yaml_document(path)
    if data is None:
        raise ValueError(f"YAML file '{path}' is empty.")
    if not isinstance(data, list):
        raise ValueError(f"YAML file '{path}' must contain a list.")

    items: list[dict[str, Any]] = []
    for index, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Entry #{index} in YAML file '{path}' must be a mapping."
            )
        items.append(entry)
    return items


def write_yaml_mapping(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def resolve_runtime_path(
    value: str | Path, runtime_root: Optional[Path] = None
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    base_dir = Path.cwd() if runtime_root is None else runtime_root
    return (base_dir / candidate).resolve()


def load_simulation_config(
    path: Path, runtime_root: Optional[Path] = None
) -> SimulationConfig:
    config_path = path.resolve()
    raw = load_yaml_mapping(config_path)

    paths = _require_mapping(raw, "paths", config_path)
    thermodynamics = _require_mapping(raw, "thermodynamics", config_path)
    system_cfg = _require_mapping(raw, "system", config_path)
    simulation_cfg = _require_mapping(raw, "simulation", config_path)
    reporting = _require_mapping(raw, "reporting", config_path)
    input_format = _resolve_input_format(paths, config_path)

    force_fields_raw = raw.get("force_fields")
    if input_format == "pdb":
        if not isinstance(force_fields_raw, list) or not force_fields_raw:
            raise ValueError(
                f"Config '{config_path}' must define a non-empty 'force_fields' list."
            )
        force_field_files = tuple(
            _require_non_empty_string(
                item, f"force_fields[{index}]", config_path
            )
            for index, item in enumerate(force_fields_raw)
        )
    else:
        if "force_fields" in raw:
            raise ValueError(
                f"Config '{config_path}' must not define 'force_fields' when "
                "using Amber input."
            )
        force_field_files = ()

    if input_format == "amber":
        if "pdb" in paths:
            raise ValueError(
                f"Config '{config_path}' must not define 'paths.pdb' when "
                "using Amber input."
            )
        if "prmtop" not in paths or "inpcrd" not in paths:
            raise ValueError(
                f"Config '{config_path}' must define both 'paths.prmtop' and "
                "'paths.inpcrd' when using Amber input."
            )
    elif "prmtop" in paths or "inpcrd" in paths:
        raise ValueError(
            f"Config '{config_path}' must not define 'paths.prmtop' or "
            "'paths.inpcrd' when using PDB input."
        )

    restraints_raw = raw.get("restraints")
    restraints = None
    if restraints_raw is not None:
        restraints_mapping = _require_mapping(
            raw, "restraints", config_path
        )
        restraints = RestraintConfig(
            force_constant_kj_per_mol_nm2=_require_non_negative_float(
                restraints_mapping.get("force_constant"),
                "restraints.force_constant",
                config_path,
            ),
            selection=_require_non_empty_string(
                restraints_mapping.get("selection"),
                "restraints.selection",
                config_path,
            ),
        )

    resolved_paths = _build_paths(
        paths=paths,
        config_path=config_path,
        runtime_root=runtime_root,
        input_format=input_format,
    )

    return SimulationConfig(
        paths=resolved_paths,
        force_field_files=force_field_files,
        thermodynamics=ThermodynamicsConfig(
            temperature_kelvin=_require_positive_float(
                thermodynamics.get("temperature"),
                "thermodynamics.temperature",
                config_path,
            ),
            pressure_bar=_require_positive_float(
                thermodynamics.get("pressure"),
                "thermodynamics.pressure",
                config_path,
            ),
            friction_coefficient_per_ps=_require_positive_float(
                thermodynamics.get("friction_coefficient"),
                "thermodynamics.friction_coefficient",
                config_path,
            ),
            step_size_ps=_require_positive_float(
                thermodynamics.get("step_size"),
                "thermodynamics.step_size",
                config_path,
            ),
        ),
        system=SystemConfig(
            nonbonded_cutoff_nm=_require_positive_float(
                system_cfg.get("nonbonded_cutoff"),
                "system.nonbonded_cutoff",
                config_path,
            ),
            solvent_padding_nm=(
                _require_positive_float(
                    system_cfg.get("solvent_padding"),
                    "system.solvent_padding",
                    config_path,
                )
                if input_format == "pdb"
                else _optional_positive_float(
                    system_cfg.get("solvent_padding"),
                    "system.solvent_padding",
                    config_path,
                )
            ),
            ionic_strength_molar=(
                _require_non_negative_float(
                    system_cfg.get("ionic_strength"),
                    "system.ionic_strength",
                    config_path,
                )
                if input_format == "pdb"
                else _optional_positive_float(
                    system_cfg.get("ionic_strength"),
                    "system.ionic_strength",
                    config_path,
                )
            ),
            hydrogen_mass_amu=_optional_positive_float(
                system_cfg.get("hydrogen_mass"),
                "system.hydrogen_mass",
                config_path,
            ),
        ),
        simulation=SimulationStages(
            nvt_steps=_require_non_negative_int(
                simulation_cfg.get("nvt_steps"),
                "simulation.nvt_steps",
                config_path,
            ),
            npt_steps=_require_non_negative_int(
                simulation_cfg.get("npt_steps"),
                "simulation.npt_steps",
                config_path,
            ),
            production_steps=_require_non_negative_int(
                simulation_cfg.get("production_steps"),
                "simulation.production_steps",
                config_path,
            ),
        ),
        reporting=ReportingConfig(
            dcd_interval=_require_positive_int(
                reporting.get("dcd_interval"),
                "reporting.dcd_interval",
                config_path,
            ),
            stdout_interval=_require_positive_int(
                reporting.get("stdout_interval"),
                "reporting.stdout_interval",
                config_path,
            ),
            log_interval=_require_positive_int(
                reporting.get("log_interval"),
                "reporting.log_interval",
                config_path,
            ),
        ),
        restraints=restraints,
    )


def _build_paths(
    *,
    paths: Mapping[str, Any],
    config_path: Path,
    runtime_root: Optional[Path],
    input_format: str,
) -> SimulationPaths:
    pdb_path: Optional[Path] = None
    prmtop_path: Optional[Path] = None
    inpcrd_path: Optional[Path] = None
    input_copy_paths: tuple[Path, ...]
    system_name: str

    if input_format == "pdb":
        pdb_path = resolve_runtime_path(
            _require_non_empty_string(paths.get("pdb"), "paths.pdb", config_path),
            runtime_root,
        )
        input_copy_paths = (pdb_path,)
        system_name = pdb_path.stem
    else:
        prmtop_path = resolve_runtime_path(
            _require_non_empty_string(
                paths.get("prmtop"), "paths.prmtop", config_path
            ),
            runtime_root,
        )
        inpcrd_path = resolve_runtime_path(
            _require_non_empty_string(
                paths.get("inpcrd"), "paths.inpcrd", config_path
            ),
            runtime_root,
        )
        input_copy_paths = (prmtop_path, inpcrd_path)
        system_name = prmtop_path.stem

    output_root = resolve_runtime_path(
        str(paths.get("output_root", "data/md_runs")),
        runtime_root,
    )
    run_id = _require_non_empty_string(
        paths.get("run_id", "default"), "paths.run_id", config_path
    )
    system_root = output_root / system_name
    run_root = system_root / run_id
    input_dir = run_root / "input"
    output_dir = run_root / "output"

    topology_path = _resolve_output_path(
        paths.get("topology"),
        "paths.topology",
        config_path,
        output_dir,
        runtime_root,
    )
    minimized_path = _resolve_output_path(
        paths.get("minimized"),
        "paths.minimized",
        config_path,
        output_dir,
        runtime_root,
    )
    trajectory_path = _resolve_output_path(
        paths.get("trajectory"),
        "paths.trajectory",
        config_path,
        output_dir,
        runtime_root,
    )
    log_path = _resolve_output_path(
        paths.get("log"),
        "paths.log",
        config_path,
        output_dir,
        runtime_root,
    )
    checkpoint_path = _resolve_output_path(
        paths.get("checkpoint", "checkpoint.chk"),
        "paths.checkpoint",
        config_path,
        output_dir,
        runtime_root,
    )

    return SimulationPaths(
        config_path=config_path,
        input_format=input_format,
        system_name=system_name,
        pdb_path=pdb_path,
        prmtop_path=prmtop_path,
        inpcrd_path=inpcrd_path,
        input_copy_paths=input_copy_paths,
        output_root=output_root,
        system_root=system_root,
        run_root=run_root,
        input_dir=input_dir,
        output_dir=output_dir,
        run_id=run_id,
        topology_path=topology_path,
        minimized_path=minimized_path,
        trajectory_path=trajectory_path,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
    )


def _resolve_input_format(
    paths: Mapping[str, Any], config_path: Path
) -> str:
    raw_format = paths.get("input_format")
    if raw_format is None:
        return _detect_legacy_input_format(paths, config_path)
    if not isinstance(raw_format, str) or not raw_format.strip():
        raise ValueError(
            f"Config '{config_path}' must define 'paths.input_format' as a non-empty string."
        )
    input_format = raw_format.strip().lower()
    if input_format not in {"pdb", "amber"}:
        raise ValueError(
            f"Config '{config_path}' must define 'paths.input_format' as 'pdb' or 'amber'."
        )
    return input_format


def _detect_legacy_input_format(
    paths: Mapping[str, Any], config_path: Path
) -> str:
    has_pdb = "pdb" in paths
    has_prmtop = "prmtop" in paths
    has_inpcrd = "inpcrd" in paths

    if has_pdb and (has_prmtop or has_inpcrd):
        raise ValueError(
            f"Config '{config_path}' must use either 'paths.pdb' or "
            "'paths.prmtop'/'paths.inpcrd', not both."
        )
    if has_pdb:
        return "pdb"
    if has_prmtop or has_inpcrd:
        if not (has_prmtop and has_inpcrd):
            raise ValueError(
                f"Config '{config_path}' must define both 'paths.prmtop' and "
                "'paths.inpcrd'."
            )
        return "amber"
    raise ValueError(
        f"Config '{config_path}' must define 'paths.input_format' or provide either "
        "'paths.pdb' or both 'paths.prmtop' and 'paths.inpcrd'."
    )


def _resolve_output_path(
    value: Any,
    field_name: str,
    config_path: Path,
    parent_dir: Path,
    runtime_root: Optional[Path],
) -> Path:
    raw_path = Path(_require_non_empty_string(value, field_name, config_path))
    if raw_path.is_absolute():
        return raw_path
    if raw_path.parent != Path("."):
        return resolve_runtime_path(raw_path, runtime_root)
    return parent_dir / raw_path


def _require_mapping(
    data: Mapping[str, Any], key: str, config_path: Path
) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise KeyError(
            f"Config '{config_path}' must define '{key}' as a mapping."
        )
    return value


def _require_non_empty_string(
    value: Any, field_name: str, config_path: Path
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Config '{config_path}' must define '{field_name}' as a non-empty string."
        )
    return value.strip()


def _require_positive_float(
    value: Any, field_name: str, config_path: Path
) -> float:
    number = _coerce_float(value, field_name, config_path)
    if number <= 0.0:
        raise ValueError(
            f"Config '{config_path}' must define '{field_name}' as a positive number."
        )
    return number


def _require_non_negative_float(
    value: Any, field_name: str, config_path: Path
) -> float:
    number = _coerce_float(value, field_name, config_path)
    if number < 0.0:
        raise ValueError(
            f"Config '{config_path}' must define '{field_name}' as a non-negative number."
        )
    return number


def _optional_positive_float(
    value: Any, field_name: str, config_path: Path
) -> Optional[float]:
    if value is None:
        return None
    return _require_positive_float(value, field_name, config_path)


def _require_positive_int(
    value: Any, field_name: str, config_path: Path
) -> int:
    number = _coerce_int(value, field_name, config_path)
    if number <= 0:
        raise ValueError(
            f"Config '{config_path}' must define '{field_name}' as a positive integer."
        )
    return number


def _require_non_negative_int(
    value: Any, field_name: str, config_path: Path
) -> int:
    number = _coerce_int(value, field_name, config_path)
    if number < 0:
        raise ValueError(
            f"Config '{config_path}' must define '{field_name}' as a non-negative integer."
        )
    return number


def _coerce_float(value: Any, field_name: str, config_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Config '{config_path}' must define '{field_name}' as a number."
        )
    return float(value)


def _coerce_int(value: Any, field_name: str, config_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Config '{config_path}' must define '{field_name}' as an integer."
        )
    return value
