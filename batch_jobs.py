from __future__ import annotations

import copy
import json
import os
import shlex
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from md_config import (
    load_simulation_config,
    load_yaml_list,
    load_yaml_mapping,
    resolve_runtime_path,
    write_yaml_mapping,
)


DEFAULT_JOBS_PATH = Path("jobs.yaml")
DEFAULT_GENERATED_DIR = Path("generated_configs")
DEFAULT_MD_SCRIPT = Path("01_md.py")
DEFAULT_PBS_TEMPLATE = Path("scheduler/pbs/md_job.pbs.j2")


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
    replicas: int
    replica_index: Optional[int]
    extra_args: tuple[str, ...]
    env: dict[str, str]
    path_overrides: dict[str, str]
    scheduler: dict[str, Any]


@dataclass(frozen=True)
class PreparedJob:
    index: int
    name: str
    command: list[str]
    env: dict[str, str]
    pbs_env: dict[str, str]
    config_path: Path
    scheduler: dict[str, Any]
    checkpoint_path: Optional[Path]
    pbs_script_path: Optional[Path] = None
    pbs_stdout_path: Optional[Path] = None
    pbs_stderr_path: Optional[Path] = None


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


def expand_job_specs(jobs: list[BatchJobSpec]) -> list[BatchJobSpec]:
    expanded: list[BatchJobSpec] = []
    for job in jobs:
        if job.replicas == 1:
            expanded.append(job)
            continue

        for replica_index in range(1, job.replicas + 1):
            expanded.append(
                replace(
                    job,
                    name=f"{job.name}-rep{replica_index:03d}",
                    replica_index=replica_index,
                )
            )
    return expanded


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
    replicas_raw = raw_job.get("replicas", 1)
    replicas = _coerce_int(replicas_raw, "replicas", index, source_path)
    if replicas <= 0:
        raise ValueError(
            f"Job #{index} in '{source_path}' must define 'replicas' as a positive integer."
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
        raw_name = _require_non_empty_string(job_name, "name", index, source_path)
        name = sanitize_component(raw_name)

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
    validate_env_names(env, index, source_path)
    path_overrides = _string_mapping(
        raw_job.get("paths", {}), "paths", index, source_path
    )
    scheduler = _mapping(
        raw_job.get("scheduler", {}), "scheduler", index, source_path
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
        replicas=replicas,
        replica_index=None,
        extra_args=extra_args,
        env=env,
        path_overrides=path_overrides,
        scheduler=scheduler,
    )


def derive_job_name(
    index: int,
    pdb: Optional[str],
    prmtop: Optional[str],
    run_id: Optional[str],
    replica_index: Optional[int] = None,
) -> str:
    source = pdb or prmtop
    assert source is not None
    source_path = Path(source)
    bits = [f"{index:03d}", sanitize_component(source_path.stem)]
    if run_id:
        bits.append(sanitize_component(run_id))
    if replica_index is not None:
        bits.append(f"rep{replica_index:03d}")
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
    expanded_jobs = expand_job_specs(jobs)
    validate_unique_job_names(expanded_jobs)

    for job in expanded_jobs:
        template_path = job.config_path if job.config_path else default_config_path
        base_config = load_base_config(template_path, base_config_cache)
        merged_config = prepare_config(base_config, job)
        written_config = write_config(job.name, generated_config_dir, merged_config)
        command = build_command(md_script, python_executable, written_config, job)
        env = os.environ.copy()
        env.update(job.env)
        pbs_env = dict(job.env)
        if job.replica_index is not None:
            env["MD_REPLICA_INDEX"] = str(job.replica_index)
            env["MD_REPLICA_TOTAL"] = str(job.replicas)
            pbs_env["MD_REPLICA_INDEX"] = str(job.replica_index)
            pbs_env["MD_REPLICA_TOTAL"] = str(job.replicas)
        prepared.append(
            PreparedJob(
                index=job.index,
                name=job.name,
                command=command,
                env=env,
                pbs_env=pbs_env,
                config_path=written_config,
                scheduler=job.scheduler,
                checkpoint_path=(
                    resolve_runtime_path(job.checkpoint)
                    if job.checkpoint is not None
                    else None
                ),
            )
        )

    validate_unique_output_paths(prepared)
    return prepared


def validate_unique_job_names(jobs: list[BatchJobSpec]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for job in jobs:
        if job.name in seen:
            duplicates.add(job.name)
        seen.add(job.name)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Expanded batch job names must be unique: {duplicate_list}")


def validate_env_names(env: dict[str, str], index: int, source_path: Path) -> None:
    invalid_names = sorted(
        name for name in env if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
    )
    if invalid_names:
        invalid_list = ", ".join(invalid_names)
        raise ValueError(
            f"Job #{index} in '{source_path}' has invalid environment variable names: "
            f"{invalid_list}"
        )


def validate_unique_output_paths(prepared_jobs: list[PreparedJob]) -> None:
    owners: dict[Path, str] = {}
    collisions: list[str] = []

    for job in prepared_jobs:
        paths = load_simulation_config(job.config_path).paths
        output_paths = (
            paths.topology_path,
            paths.minimized_path,
            paths.trajectory_path,
            paths.log_path,
            job.checkpoint_path or paths.checkpoint_path,
        )
        for output_path in output_paths:
            resolved_path = output_path.resolve()
            owner = owners.get(resolved_path)
            if owner is not None:
                collisions.append(f"{resolved_path} ({owner}, {job.name})")
            else:
                owners[resolved_path] = job.name

    if collisions:
        collision_list = "; ".join(collisions)
        raise ValueError(f"Batch jobs must not share output paths: {collision_list}")


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

    if job.replica_index is not None:
        base_run_id = str(paths.get("run_id", f"job_{job.index:03d}"))
        paths["run_id"] = f"{base_run_id}_rep{job.replica_index:03d}"

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


def submit_pbs_jobs(
    prepared_jobs: list[PreparedJob],
    submission_dir: Path,
    pbs_template_path: Path,
    dry_run: bool,
    qsub_command: str = "qsub",
) -> None:
    if shutil.which(qsub_command) is None and not dry_run:
        raise FileNotFoundError(
            f"PBS submit command '{qsub_command}' was not found on PATH."
        )

    scheduler = resolve_common_scheduler(prepared_jobs)
    task_script_dir = submission_dir / "tasks"
    submission_dir.mkdir(parents=True, exist_ok=True)
    task_script_dir.mkdir(parents=True, exist_ok=True)

    task_manifest_path = submission_dir / "task_manifest.txt"
    task_script_paths: list[Path] = []

    for job in prepared_jobs:
        task_script_path = task_script_dir / f"{job.name}.sh"
        task_stdout_path = task_script_dir / f"{job.name}.out"
        task_stderr_path = task_script_dir / f"{job.name}.err"
        task_script_path.write_text(
            render_task_script(
                job=job,
                stdout_path=task_stdout_path,
                stderr_path=task_stderr_path,
            )
        )
        os.chmod(task_script_path, 0o755)
        task_script_paths.append(task_script_path)
        print(f"[job {job.index:03d} | {job.name}] task script: {task_script_path}")

    task_manifest_path.write_text(
        "".join(f"{path.as_posix()}\n" for path in task_script_paths)
    )

    array_script_path = submission_dir / "batch_array.pbs"
    array_script_path.write_text(
        render_array_pbs_script(
            template=pbs_template_path.read_text(),
            scheduler=scheduler,
            job_count=len(prepared_jobs),
            manifest_path=task_manifest_path,
            job_name="batch_md_array",
            stdout_path=submission_dir / "batch_array.out",
            stderr_path=submission_dir / "batch_array.err",
        )
    )
    os.chmod(array_script_path, 0o755)

    print(f"[pbs] array script: {array_script_path}")
    print(f"[pbs] manifest: {task_manifest_path}")
    print(f"[pbs] qsub command: {qsub_command} {array_script_path}")
    if dry_run:
        return

    completed = subprocess.run(
        [qsub_command, str(array_script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(completed.returncode)
    if completed.stdout.strip():
        print(f"[pbs] qsub response: {completed.stdout.strip()}")


def render_task_script(
    *,
    job: PreparedJob,
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    env_lines = "".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in job.pbs_env.items()
    )
    command = shlex.join(job.command)
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n\n"
        f"exec > {shlex.quote(str(stdout_path))} 2> {shlex.quote(str(stderr_path))}\n\n"
        f"printf '[%s] starting job %s\\n' \"$(date -Iseconds)\" {shlex.quote(job.name)}\n"
        f"printf 'Working directory: %s\\n' {shlex.quote(str(Path.cwd()))}\n\n"
        f"cd {shlex.quote(str(Path.cwd()))}\n\n"
        f"{env_lines}"
        f"printf '[%s] launching: %s\\n' \"$(date -Iseconds)\" {shlex.quote(command)}\n"
        f"{command}\n\n"
        f"printf '[%s] job %s finished\\n' \"$(date -Iseconds)\" {shlex.quote(job.name)}\n"
    )


def render_array_pbs_script(
    *,
    template: str,
    scheduler: dict[str, Any],
    job_count: int,
    manifest_path: Path,
    job_name: str,
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    queue_directive = _pbs_directive("q", scheduler.get("queue"))
    account_directive = _pbs_directive("A", scheduler.get("account"))
    walltime = scheduler.get("walltime")
    walltime_directive = (
        f"#PBS -l walltime={walltime}\n"
        if walltime is not None and str(walltime).strip()
        else ""
    )
    resource_directive = _pbs_resource_directive(scheduler.get("resources"))
    array_directive = _pbs_array_directive(
        job_count=job_count,
        array_flag=scheduler.get("array_flag", "J"),
        max_concurrent=scheduler.get("max_concurrent"),
        submit_flags=scheduler.get("submit_flags"),
    )
    return template.format(
        job_name=job_name,
        queue_directive=queue_directive,
        account_directive=account_directive,
        resource_directive=resource_directive,
        walltime_directive=walltime_directive,
        array_directive=array_directive,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        workdir=shlex.quote(str(Path.cwd())),
        manifest_path=shlex.quote(str(manifest_path)),
        module_lines=_lines_block(scheduler.get("module_lines")),
        pre_command_lines=_lines_block(scheduler.get("pre_command_lines")),
    )


def _pbs_directive(flag: str, value: Optional[str]) -> str:
    if value is None or not str(value).strip():
        return ""
    return f"#PBS -{flag} {value}\n"


def _pbs_resource_directive(resources: Any) -> str:
    if resources is None:
        return ""
    if isinstance(resources, str) and resources.strip():
        return f"#PBS -l {resources.strip()}\n"
    if not isinstance(resources, dict) or not resources:
        return ""
    bits = []
    for key, value in resources.items():
        bits.append(f"{key}={value}")
    return "#PBS -l " + ":".join(bits) + "\n"


def _lines_block(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value if value.endswith("\n") else value + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            lines.append(str(item))
        return "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    return ""


def _pbs_submit_flags(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [str(item) for item in value]
    else:
        return ""

    lines = []
    for item in items:
        stripped = item.strip()
        if not stripped:
            continue
        if stripped.startswith("#PBS"):
            lines.append(stripped if stripped.endswith("\n") else stripped + "\n")
        elif stripped.startswith("-"):
            lines.append(f"#PBS {stripped}\n")
        else:
            lines.append(f"#PBS -{stripped}\n")
    return "".join(lines)


def _pbs_array_directive(
    *,
    job_count: int,
    array_flag: Any,
    max_concurrent: Any,
    submit_flags: Any,
) -> str:
    if array_flag not in {"J", "t"}:
        raise ValueError("scheduler.array_flag must be 'J' or 't'.")

    array_range = f"1-{job_count}"
    if max_concurrent is not None:
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or max_concurrent <= 0
        ):
            raise ValueError("scheduler.max_concurrent must be a positive integer.")
        array_range += f"%{max_concurrent}"

    directive = f"#PBS -{array_flag} {array_range}\n"
    return directive + _pbs_submit_flags(submit_flags)


def resolve_common_scheduler(prepared_jobs: list[PreparedJob]) -> dict[str, Any]:
    if not prepared_jobs:
        raise ValueError("At least one job is required to build a PBS array.")

    base_scheduler = prepared_jobs[0].scheduler
    for job in prepared_jobs[1:]:
        if _scheduler_signature(job.scheduler) != _scheduler_signature(base_scheduler):
            raise ValueError(
                "PBS array mode requires all jobs to share the same scheduler settings."
            )
    return base_scheduler


def _scheduler_signature(scheduler: dict[str, Any]) -> str:
    return json.dumps(scheduler, sort_keys=True, separators=(",", ":"), default=str)


def run_batch_jobs(
    *,
    jobs_path: Path,
    default_config_path: Path,
    generated_config_dir: Path,
    md_script: Path,
    python_executable: str,
    workers: int,
    mode: str,
    pbs_template_path: Path,
    qsub_command: str,
    dry_run: bool,
) -> None:
    job_specs = load_job_specs(jobs_path, default_config_path)
    submission_dir: Optional[Path] = None
    effective_generated_config_dir = generated_config_dir
    if mode == "pbs":
        submission_dir = create_pbs_submission_dir(generated_config_dir)
        effective_generated_config_dir = submission_dir / "configs"

    prepared_jobs = prepare_jobs(
        jobs=job_specs,
        default_config_path=default_config_path,
        generated_config_dir=effective_generated_config_dir,
        md_script=md_script,
        python_executable=python_executable,
    )
    if mode == "pbs":
        assert submission_dir is not None
        submit_pbs_jobs(
            prepared_jobs,
            submission_dir=submission_dir,
            pbs_template_path=pbs_template_path,
            dry_run=dry_run,
            qsub_command=qsub_command,
        )
        return
    run_jobs(prepared_jobs, workers, dry_run)


def create_pbs_submission_dir(generated_config_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_id = f"{timestamp}-{uuid4().hex[:8]}"
    return generated_config_dir / "pbs_submissions" / submission_id


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


def _mapping(
    value: Any, field_name: str, index: int, source_path: Path
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"Job #{index} in '{source_path}' must define '{field_name}' as a mapping."
        )
    return dict(value)


def _coerce_float(
    value: Any, field_name: str, index: int, source_path: Path
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Job #{index} in '{source_path}' must define '{field_name}' as a number."
    )
    return float(value)


def _coerce_int(value: Any, field_name: str, index: int, source_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Job #{index} in '{source_path}' must define '{field_name}' as an integer."
        )
    return value


def _coerce_path(
    value: Any, field_name: str, index: int, source_path: Path
) -> Path:
    if isinstance(value, Path):
        return value
    return Path(_require_non_empty_string(value, field_name, index, source_path))
