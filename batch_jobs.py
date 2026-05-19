from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from md_config import load_yaml_list, load_yaml_mapping, write_yaml_mapping


DEFAULT_JOBS_PATH = Path("jobs.yaml")
DEFAULT_GENERATED_DIR = Path("generated_configs")
DEFAULT_MD_SCRIPT = Path("01_md.py")


@dataclass(frozen=True)
class BatchJobSpec:
    index: int
    name: str
    pdb: Optional[str]
    prmtop: Optional[str]
    inpcrd: Optional[str]
    config_path: Path
    run_id: Optional[str]
    output_root: Optional[str]
    until_ns: Optional[float]
    checkpoint: Optional[str]
    restart: bool
    extra_args: tuple[str, ...]
    env: dict[str, str]
    path_overrides: dict[str, str]


@dataclass(frozen=True)
class PreparedJob:
    index: int
    name: str
    command: list[str]
    env: dict[str, str]
    config_path: Path


def load_job_specs(path: Path, default_config_path: Path) -> list[BatchJobSpec]:
    raw_jobs = load_yaml_list(path)
    return [
        parse_job_spec(
            index=index,
            raw_job=raw_job,
            source_path=path,
            default_config_path=default_config_path,
        )
        for index, raw_job in enumerate(raw_jobs, start=1)
    ]


def parse_job_spec(
    *,
    index: int,
    raw_job: dict[str, Any],
    source_path: Path,
    default_config_path: Path,
) -> BatchJobSpec:
    pdb = _optional_string(raw_job.get("pdb"), "pdb", index, source_path)
    prmtop = _optional_string(raw_job.get("prmtop"), "prmtop", index, source_path)
    inpcrd = _optional_string(raw_job.get("inpcrd"), "inpcrd", index, source_path)
    if pdb is None and prmtop is None and inpcrd is None:
        raise ValueError(
            f"Job #{index} in '{source_path}' must define either 'pdb' or both "
            "'prmtop' and 'inpcrd'."
        )
    if pdb is not None and (prmtop is not None or inpcrd is not None):
        raise ValueError(
            f"Job #{index} in '{source_path}' must not mix 'pdb' with "
            "'prmtop'/'inpcrd'."
        )
    if (prmtop is None) != (inpcrd is None):
        raise ValueError(
            f"Job #{index} in '{source_path}' must define both 'prmtop' and 'inpcrd'."
        )
    run_id = _optional_string(raw_job.get("run_id"), "run_id", index, source_path)
    output_root = _optional_string(
        raw_job.get("output_root"), "output_root", index, source_path
    )
    checkpoint = _optional_string(
        raw_job.get("checkpoint"), "checkpoint", index, source_path
    )
    job_name = raw_job.get("name")
    if job_name is None:
        name = derive_job_name(index, pdb, prmtop, run_id)
    else:
        name = _require_non_empty_string(job_name, "name", index, source_path)

    config_path_raw = raw_job.get("config", default_config_path)
    config_path = _coerce_path(config_path_raw, "config", index, source_path)

    until_ns_raw = raw_job.get("until_ns")
    until_ns = None
    if until_ns_raw is not None:
        until_ns = _coerce_float(until_ns_raw, "until_ns", index, source_path)

    extra_args_raw = raw_job.get("extra_args", [])
    if not isinstance(extra_args_raw, list):
        raise ValueError(
            f"Job #{index} in '{source_path}' must define 'extra_args' as a list."
        )
    extra_args = tuple(str(arg) for arg in extra_args_raw)

    env = _string_mapping(raw_job.get("env", {}), "env", index, source_path)
    path_overrides = _string_mapping(
        raw_job.get("paths", {}), "paths", index, source_path
    )

    restart = raw_job.get("restart", False)
    if not isinstance(restart, bool):
        raise ValueError(
            f"Job #{index} in '{source_path}' must define 'restart' as a boolean."
        )

    return BatchJobSpec(
        index=index,
        name=name,
        pdb=pdb,
        prmtop=prmtop,
        inpcrd=inpcrd,
        config_path=config_path,
        run_id=run_id,
        output_root=output_root,
        until_ns=until_ns,
        checkpoint=checkpoint,
        restart=restart,
        extra_args=extra_args,
        env=env,
        path_overrides=path_overrides,
    )


def derive_job_name(
    index: int, pdb: Optional[str], prmtop: Optional[str], run_id: Optional[str]
) -> str:
    source = pdb or prmtop
    assert source is not None
    source_path = Path(source)
    bits = [f"{index:03d}", sanitize_component(source_path.stem)]
    if run_id:
        bits.append(sanitize_component(run_id))
    return "-".join(bits)


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("_") or "run"


def prepare_jobs(
    *,
    jobs: list[BatchJobSpec],
    default_config_path: Path,
    generated_config_dir: Path,
    md_script: Path,
    python_executable: str,
) -> list[PreparedJob]:
    base_config_cache: dict[Path, dict[str, Any]] = {}
    prepared: list[PreparedJob] = []

    for job in jobs:
        template_path = job.config_path if job.config_path else default_config_path
        base_config = load_base_config(template_path, base_config_cache)
        merged_config = prepare_config(base_config, job)
        written_config = write_config(job.name, generated_config_dir, merged_config)
        command = build_command(md_script, python_executable, written_config, job)
        env = os.environ.copy()
        env.update(job.env)
        prepared.append(
            PreparedJob(
                index=job.index,
                name=job.name,
                command=command,
                env=env,
                config_path=written_config,
            )
        )

    return prepared


def load_base_config(
    path: Path, cache: dict[Path, dict[str, Any]]
) -> dict[str, Any]:
    if path not in cache:
        cache[path] = load_yaml_mapping(path)
    return cache[path]


def prepare_config(
    base_config: dict[str, Any], job: BatchJobSpec
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("The base config must expose 'paths' as a mapping.")

    if job.pdb is not None:
        paths["pdb"] = job.pdb
        paths.pop("prmtop", None)
        paths.pop("inpcrd", None)
    elif job.prmtop is not None and job.inpcrd is not None:
        paths["prmtop"] = job.prmtop
        paths["inpcrd"] = job.inpcrd
        paths.pop("pdb", None)
    if job.output_root is not None:
        paths["output_root"] = job.output_root
    if job.run_id is not None:
        paths["run_id"] = job.run_id
    elif "run_id" not in paths:
        paths["run_id"] = f"job_{job.index:03d}"

    for extra_key, extra_value in job.path_overrides.items():
        paths[extra_key] = extra_value

    return config


def write_config(
    job_name: str, generated_dir: Path, config: dict[str, Any]
) -> Path:
    config_path = generated_dir / f"{job_name}.yaml"
    write_yaml_mapping(config_path, config)
    return config_path


def build_command(
    md_script: Path,
    python_executable: str,
    config_path: Path,
    job: BatchJobSpec,
) -> list[str]:
    command = [python_executable, str(md_script), "--config", str(config_path)]
    if job.checkpoint is not None:
        command.extend(["--checkpoint", job.checkpoint])
    if job.restart:
        command.append("--restart")
    if job.until_ns is not None:
        command.extend(["--until", str(job.until_ns)])
    command.extend(job.extra_args)
    return command


def execute_job(job: PreparedJob, dry_run: bool) -> int:
    cmd_str = " ".join(job.command)
    prefix = f"[job {job.index:03d} | {job.name}]"
    print(f"{prefix} config: {job.config_path}")
    print(f"{prefix} command: {cmd_str}")
    if dry_run:
        return 0

    completed = subprocess.run(job.command, env=job.env)
    if completed.returncode != 0:
        print(
            f"{prefix} failed with exit code {completed.returncode}",
            file=sys.stderr,
        )
    else:
        print(f"{prefix} completed successfully")
    return completed.returncode


def run_jobs(prepared_jobs: list[PreparedJob], workers: int, dry_run: bool) -> None:
    if workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    if workers == 1:
        for job in prepared_jobs:
            code = execute_job(job, dry_run)
            if code != 0:
                raise SystemExit(code)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(execute_job, job, dry_run): job for job in prepared_jobs
        }
        for future in as_completed(futures):
            code = future.result()
            if code != 0:
                raise SystemExit(code)


def run_batch_jobs(
    *,
    jobs_path: Path,
    default_config_path: Path,
    generated_config_dir: Path,
    md_script: Path,
    python_executable: str,
    workers: int,
    dry_run: bool,
) -> None:
    job_specs = load_job_specs(jobs_path, default_config_path)
    prepared_jobs = prepare_jobs(
        jobs=job_specs,
        default_config_path=default_config_path,
        generated_config_dir=generated_config_dir,
        md_script=md_script,
        python_executable=python_executable,
    )
    run_jobs(prepared_jobs, workers, dry_run)


def _require_non_empty_string(
    value: Any, field_name: str, index: int, source_path: Path
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Job #{index} in '{source_path}' must define '{field_name}' as a non-empty string."
        )
    return value.strip()


def _optional_string(
    value: Any, field_name: str, index: int, source_path: Path
) -> Optional[str]:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name, index, source_path)


def _string_mapping(
    value: Any, field_name: str, index: int, source_path: Path
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(
            f"Job #{index} in '{source_path}' must define '{field_name}' as a mapping."
        )
    return {str(key): str(item) for key, item in value.items()}


def _coerce_float(
    value: Any, field_name: str, index: int, source_path: Path
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Job #{index} in '{source_path}' must define '{field_name}' as a number."
        )
    return float(value)


def _coerce_path(
    value: Any, field_name: str, index: int, source_path: Path
) -> Path:
    if isinstance(value, Path):
        return value
    return Path(_require_non_empty_string(value, field_name, index, source_path))
