from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

"""
Example jobs.yaml:

- pdb: ../biomolecular_model_systems/peptides/small/diala.pdb
  run_id: replica_00
  until_ns: 100
  env:
    CUDA_VISIBLE_DEVICES: "0"
- pdb: ../biomolecular_model_systems/peptides/small/alanine.pdb
  run_id: replica_01
  checkpoint: checkpoints/alanine.chk
  restart: true

Each entry at minimum needs a 'pdb'. Optional keys include:
  run_id       -> overrides paths.run_id
  output_root  -> overrides paths.output_root
  config       -> base config template (defaults to --default-config)
  until_ns     -> forwarded to 01_md.py --until
  checkpoint   -> forwarded to --checkpoint
  restart      -> bool flag forwarded to --restart
  extra_args   -> list of additional CLI arguments for 01_md.py
  env          -> mapping of environment variables to add for the job
  paths        -> mapping merged into config['paths'] for advanced overrides
"""

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency PyYAML. Install with `pip install pyyaml`."
    ) from exc


DEFAULT_JOBS_PATH = Path("jobs.yaml")
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_GENERATED_DIR = Path("generated_configs")
DEFAULT_MD_SCRIPT = Path("01_md.py")
DEFAULT_SUBMISSION_DIR = Path("scheduler/pbs/generated")
DEFAULT_LOG_DIR = Path("scheduler/pbs/logs")


@dataclass(frozen=True)
class PreparedJob:
    index: int
    name: str
    command: List[str]
    env: Dict[str, str]
    config_path: Path
    replica_id: int


@dataclass(frozen=True)
class JobBundle:
    index: int
    name: str
    config_path: Path
    scheduler: Dict[str, Any]
    env: Dict[str, str]
    jobs: List[PreparedJob]
    meta: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple OpenMM runs driven by a jobs YAML file."
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        default=DEFAULT_JOBS_PATH,
        help=f"YAML file describing the runs (default: {DEFAULT_JOBS_PATH}).",
    )
    parser.add_argument(
        "--default-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Base config used when jobs do not specify one.",
    )
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Directory to write the per-job config files into.",
    )
    parser.add_argument(
        "--md-script",
        type=Path,
        default=DEFAULT_MD_SCRIPT,
        help="Path to the MD driver script (default: 01_md.py).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help=f"Python interpreter used to launch jobs (default: {sys.executable}).",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "pbs"),
        default="local",
        help="Execution mode: 'local' runs jobs directly, 'pbs' generates qsub scripts.",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=DEFAULT_SUBMISSION_DIR,
        help=(
            "Directory for generated PBS scripts (PBS mode; default: "
            f"{DEFAULT_SUBMISSION_DIR})."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=(
            "Directory for PBS stdout/stderr logs (PBS mode; default: "
            f"{DEFAULT_LOG_DIR})."
        ),
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Override scheduler template path (defaults to scheduler.template in config).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit generated PBS scripts with qsub (PBS mode only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of jobs to run concurrently. Use 1 for sequential execution.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help=(
            "CSV path for job manifest (default: generated_configs/jobs_manifest.csv)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands that would be executed.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Any:
    try:
        with path.open() as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError:
        raise SystemExit(f"YAML file {path} not found.")


def ensure_list(data: Any, path: Path) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of jobs in {path}, got {type(data)}.")
    jobs: List[Dict[str, Any]] = []
    for idx, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f"Job #{idx} in {path} is not a mapping.")
        jobs.append(entry)
    return jobs


def deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def directive_line(option: str, value: Optional[str]) -> str:
    if not value:
        return ""
    return f"#PBS {option} {value}\n"


def format_pbs_resource_line(resources: Dict[str, Any]) -> str:
    if not resources:
        return ""
    entries: Dict[str, Any] = {str(k): v for k, v in resources.items()}
    nodes = entries.pop("nodes", entries.pop("select", 1))
    parts = [f"select={nodes}"]
    if "ppn" in entries:
        entries["ncpus"] = entries.pop("ppn")
    key_order = ["ncpus", "ngpus", "gpus", "mem"]
    for key in key_order:
        if key in entries:
            parts.append(f"{key}={entries.pop(key)}")
    for key, value in entries.items():
        parts.append(f"{key}={value}")
    return ":".join(str(part) for part in parts)


def render_shell_lines(items: Iterable[str]) -> str:
    lines = [str(item) for item in items if str(item).strip()]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def render_env_exports(env: Dict[str, str]) -> str:
    if not env:
        return ""
    lines = [
        f'export {key}={shlex.quote(str(value))}' for key, value in env.items()
    ]
    return "\n".join(lines) + "\n"


def escape_braces(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def append_line(block: str, line: str) -> str:
    if not line:
        return block
    if not block:
        return line if line.endswith("\n") else line + "\n"
    if not block.endswith("\n"):
        block += "\n"
    return block + (line if line.endswith("\n") else line + "\n")


class SafeDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - fallback
        return ""


def render_template(template_path: Path, context: Dict[str, str]) -> str:
    template = template_path.read_text()
    return template.format_map(SafeDict(context))


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("_") or "run"


def derive_job_name(index: int, job: Dict[str, Any]) -> str:
    if "name" in job:
        return str(job["name"])
    pdb_path = Path(str(job.get("pdb", f"job_{index}")))
    run_id = job.get("run_id")
    bits = [f"{index:03d}", sanitize_component(pdb_path.stem)]
    if run_id:
        bits.append(sanitize_component(str(run_id)))
    return "-".join(bits)


def load_base_config(path: Path, cache: Dict[Path, Dict[str, Any]]) -> Dict[str, Any]:
    if path not in cache:
        data = load_yaml(path)
        if not isinstance(data, dict):
            raise SystemExit(f"Config file {path} must contain a mapping.")
        cache[path] = data
    return cache[path]


def prepare_config(
    base_config: Dict[str, Any],
    job: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise SystemExit("The base config must expose a 'paths' mapping.")

    pdb_path = job.get("pdb")
    if not pdb_path:
        raise SystemExit(f"Job #{index} is missing the required 'pdb' field.")
    paths["pdb"] = str(pdb_path)

    if "output_root" in job:
        paths["output_root"] = str(job["output_root"])
    if "run_id" in job:
        paths["run_id"] = str(job["run_id"])
    elif "run_id" not in paths:
        paths["run_id"] = f"job_{index:03d}"

    for extra_key, extra_value in job.get("paths", {}).items():
        paths[extra_key] = str(extra_value)

    if "replicas" in job:
        replicas_cfg = config.setdefault("replicas", {})
        if not isinstance(replicas_cfg, dict):
            raise SystemExit("Config 'replicas' section must be a mapping.")
        replicas_value = job["replicas"]
        if isinstance(replicas_value, int):
            replicas_cfg["count"] = int(replicas_value)
        elif isinstance(replicas_value, dict):
            deep_update(replicas_cfg, replicas_value)
        else:
            raise SystemExit(
                f"'replicas' override for job #{index} must be int or mapping."
            )

    scheduler_override = job.get("scheduler")
    if scheduler_override:
        if not isinstance(scheduler_override, dict):
            raise SystemExit(
                f"'scheduler' override for job #{index} must be a mapping."
            )
        scheduler_cfg = config.setdefault("scheduler", {})
        if not isinstance(scheduler_cfg, dict):
            raise SystemExit("Config 'scheduler' section must be a mapping.")
        deep_update(scheduler_cfg, scheduler_override)

    return config


def write_config(
    job_name: str, generated_dir: Path, config: Dict[str, Any]
) -> Path:
    generated_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_dir / f"{job_name}.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def build_command(
    md_script: Path,
    python_exe: str,
    config_path: Path,
    job: Dict[str, Any],
    replica_id: int,
    scheduler_flags: Iterable[str],
) -> List[str]:
    command = [
        python_exe,
        str(md_script),
        "--config",
        str(config_path),
        "--replica-id",
        str(replica_id),
    ]
    for flag in scheduler_flags:
        command.append(str(flag))
    checkpoint = job.get("checkpoint")
    if checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])
    if job.get("restart"):
        command.append("--restart")
    until_ns = job.get("until_ns")
    if until_ns is not None:
        command.extend(["--until", str(until_ns)])
    stage = job.get("stage")
    if stage:
        command.extend(["--stage", str(stage)])
    seed = job.get("seed")
    if seed is not None:
        command.extend(["--seed", str(seed)])
    for arg in job.get("extra_args", []):
        command.append(str(arg))
    return command


def prepare_jobs(
    args: argparse.Namespace,
    jobs_data: Iterable[Dict[str, Any]],
) -> List[JobBundle]:
    base_config_cache: Dict[Path, Dict[str, Any]] = {}
    bundles: List[JobBundle] = []
    for index, job in enumerate(jobs_data, start=1):
        job_name = derive_job_name(index, job)
        config_path = Path(job.get("config", args.default_config))
        base_config = load_base_config(config_path, base_config_cache)
        merged_config = prepare_config(base_config, job, index)
        written_config = write_config(
            job_name, args.generated_config_dir, merged_config
        )

        scheduler_cfg = merged_config.get("scheduler", {}) or {}
        if not isinstance(scheduler_cfg, dict):
            raise SystemExit("Config 'scheduler' section must be a mapping.")
        scheduler_flags = scheduler_cfg.get("additional_flags", [])
        if scheduler_flags is None:
            scheduler_flags = []
        if not isinstance(scheduler_flags, Iterable) or isinstance(
            scheduler_flags, (str, bytes)
        ):
            raise SystemExit(
                "'scheduler.additional_flags' must be a list of CLI arguments."
            )

        replicas_cfg = merged_config.get("replicas", {}) or {}
        if not isinstance(replicas_cfg, dict):
            raise SystemExit("Config 'replicas' section must be a mapping.")
        replica_count = int(replicas_cfg.get("count", 1))
        if replica_count <= 0:
            raise SystemExit("Replica count must be a positive integer.")

        job_env_overrides = {
            str(key): str(value) for key, value in job.get("env", {}).items()
        }

        prepared_jobs: List[PreparedJob] = []
        for replica_id in range(replica_count):
            replica_label = f"{job_name}-rep{replica_id:03d}"
            command = build_command(
                args.md_script,
                scheduler_cfg.get("python_exe", args.python),
                written_config,
                job,
                replica_id,
                scheduler_flags,
            )
            env = os.environ.copy()
            env.update(job_env_overrides)
            prepared_jobs.append(
                PreparedJob(
                    index=index,
                    name=replica_label,
                    command=command,
                    env=env,
                    config_path=written_config,
                    replica_id=replica_id,
                )
            )

        bundles.append(
            JobBundle(
                index=index,
                name=job_name,
                config_path=written_config,
                scheduler=scheduler_cfg,
                env=job_env_overrides,
                jobs=prepared_jobs,
                meta=job,
            )
        )
    return bundles


def execute_job(job: PreparedJob, dry_run: bool) -> int:
    cmd_str = shlex.join(job.command)
    prefix = f"[job {job.index:03d} | {job.name}]"
    print(f"{prefix} config: {job.config_path}")
    print(f"{prefix} command: {cmd_str}")
    if dry_run:
        return 0
    completed = subprocess.run(job.command, env=job.env)
    if completed.returncode != 0:
        print(f"{prefix} failed with exit code {completed.returncode}", file=sys.stderr)
    else:
        print(f"{prefix} completed successfully")
    return completed.returncode


def run_local_jobs(
    bundles: List[JobBundle], workers: int, dry_run: bool
) -> None:
    prepared = [job for bundle in bundles for job in bundle.jobs]
    if not prepared:
        print("No jobs prepared.")
        return
    if workers <= 1:
        for job in prepared:
            code = execute_job(job, dry_run)
            if code != 0:
                raise SystemExit(code)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(execute_job, job, dry_run): job for job in prepared
        }
        for future in as_completed(futures):
            code = future.result()
            if code != 0:
                raise SystemExit(code)


def submit_job(
    script_path: Path,
    submit_command: str,
    submit_flags: Iterable[str],
    dry_run: bool,
) -> None:
    cmd = [submit_command, *map(str, submit_flags), str(script_path)]
    print(f"[pbs] submit {' '.join(cmd)}")
    if dry_run:
        return
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(
            f"qsub submission failed for {script_path} "
            f"(exit code {completed.returncode})."
        )


def generate_pbs_scripts(
    bundles: List[JobBundle],
    args: argparse.Namespace,
) -> Dict[Tuple[str, int], Path]:
    script_map: Dict[Tuple[str, int], Path] = {}
    submission_dir = args.submission_dir.resolve()
    logs_root = args.log_dir.resolve()
    submission_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    for bundle in bundles:
        scheduler_cfg = bundle.scheduler or {}
        template_value = args.template or scheduler_cfg.get("template")
        if not template_value:
            raise SystemExit(
                "Scheduler template not defined. Set 'scheduler.template' in the "
                f"config or provide --template (job '{bundle.name}')."
            )
        template_path = Path(template_value)
        if not template_path.is_absolute():
            template_path = (Path.cwd() / template_path).resolve()
        if not template_path.exists():
            raise SystemExit(f"Scheduler template '{template_path}' not found.")

        environment_cfg = scheduler_cfg.get("environment", {}) or {}
        module_lines = render_shell_lines(environment_cfg.get("modules", []))
        pre_command_lines = render_shell_lines(environment_cfg.get("pre_commands", []))
        env_lines = render_env_exports(bundle.env)

        job_log_dir = (logs_root / bundle.name).resolve()
        job_log_dir.mkdir(parents=True, exist_ok=True)

        workdir = Path(
            bundle.meta.get("workdir")
            or scheduler_cfg.get("workdir")
            or os.getcwd()
        ).resolve()

        queue_directive = directive_line("-q", scheduler_cfg.get("queue"))
        account_directive = directive_line("-A", scheduler_cfg.get("account"))
        walltime_value = scheduler_cfg.get("walltime")
        walltime_directive = directive_line(
            "-l", f"walltime={walltime_value}" if walltime_value else None
        )
        resource_line = format_pbs_resource_line(
            scheduler_cfg.get("resources", {}) or {}
        )
        resource_directive = (
            directive_line("-l", resource_line) if resource_line else ""
        )

        submit_command = scheduler_cfg.get("submit_command", "qsub")
        submit_flags = [str(flag) for flag in scheduler_cfg.get("submit_flags", [])]

        array_cfg = scheduler_cfg.get("array", {}) or {}
        use_array = array_cfg.get("enabled", False) and len(bundle.jobs) > 1
        array_chunk = array_cfg.get("chunk_size")

        if use_array:
            array_range = f"0-{len(bundle.jobs) - 1}"
            if array_chunk:
                array_range = f"{array_range}%{array_chunk}"
            array_directive = directive_line("-t", array_range)
            command = bundle.jobs[0].command.copy()
            try:
                idx = command.index("--replica-id")
                command[idx + 1] = "${PBS_ARRAY_INDEX}"
            except ValueError:
                command.extend(["--replica-id", "${PBS_ARRAY_INDEX}"])
            command_str = escape_braces(shlex.join(command))
            env_lines_array = append_line(
                env_lines, "export REPLICA_ID=${PBS_ARRAY_INDEX}"
            )
            stdout_path = escape_braces(
                str(job_log_dir / f"{bundle.name}.o.${{PBS_ARRAY_INDEX}}")
            )
            stderr_path = escape_braces(
                str(job_log_dir / f"{bundle.name}.e.${{PBS_ARRAY_INDEX}}")
            )
            context = {
                "job_name": sanitize_component(bundle.name),
                "queue_directive": queue_directive,
                "account_directive": account_directive,
                "resource_directive": resource_directive,
                "walltime_directive": walltime_directive,
                "array_directive": array_directive,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "module_lines": escape_braces(module_lines),
                "pre_command_lines": escape_braces(pre_command_lines),
                "env_lines": escape_braces(env_lines_array),
                "workdir": escape_braces(str(workdir)),
                "command": command_str,
            }
            script_path = submission_dir / f"{bundle.name}.pbs"
            script_text = render_template(template_path, context)
            script_path.write_text(script_text)
            print(f"[pbs] wrote {script_path}")
            if args.submit:
                submit_job(script_path, submit_command, submit_flags, args.dry_run)
            for job in bundle.jobs:
                script_map[(bundle.name, job.replica_id)] = script_path
            continue

        # Non-array: one script per replica
        for job in bundle.jobs:
            command_str = escape_braces(shlex.join(job.command))
            env_lines_job = append_line(
                env_lines, f"export REPLICA_ID={job.replica_id}"
            )
            stdout_path = escape_braces(
                str(job_log_dir / f"{job.name}.out")
            )
            stderr_path = escape_braces(
                str(job_log_dir / f"{job.name}.err")
            )
            context = {
                "job_name": sanitize_component(job.name),
                "queue_directive": queue_directive,
                "account_directive": account_directive,
                "resource_directive": resource_directive,
                "walltime_directive": walltime_directive,
                "array_directive": "",
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "module_lines": escape_braces(module_lines),
                "pre_command_lines": escape_braces(pre_command_lines),
                "env_lines": escape_braces(env_lines_job),
                "workdir": escape_braces(str(workdir)),
                "command": command_str,
            }
            script_path = submission_dir / f"{job.name}.pbs"
            script_text = render_template(template_path, context)
            script_path.write_text(script_text)
            print(f"[pbs] wrote {script_path}")
            if args.submit:
                submit_job(script_path, submit_command, submit_flags, args.dry_run)
            script_map[(bundle.name, job.replica_id)] = script_path

    return script_map


def write_manifest(
    bundles: List[JobBundle],
    manifest_path: Path,
    mode: str,
    script_map: Optional[Dict[Tuple[str, int], Path]] = None,
) -> None:
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_index",
                "job_name",
                "replica_id",
                "config_path",
                "command",
                "mode",
                "script_path",
            ],
        )
        writer.writeheader()
        for bundle in bundles:
            for job in bundle.jobs:
                script_entry = ""
                if script_map is not None:
                    script_entry = str(
                        script_map.get((bundle.name, job.replica_id), "")
                    )
                writer.writerow(
                    {
                        "job_index": bundle.index,
                        "job_name": bundle.name,
                        "replica_id": job.replica_id,
                        "config_path": str(job.config_path),
                        "command": shlex.join(job.command),
                        "mode": mode,
                        "script_path": script_entry,
                    }
                )
    print(f"[info] wrote manifest to {manifest_path}")


def main() -> None:
    args = parse_args()
    jobs_data = ensure_list(load_yaml(args.jobs), args.jobs)
    bundles = prepare_jobs(args, jobs_data)
    script_map: Optional[Dict[Tuple[str, int], Path]] = None
    if args.mode == "pbs":
        script_map = generate_pbs_scripts(bundles, args)
    else:
        run_local_jobs(bundles, args.workers, args.dry_run)

    manifest_path = (
        args.manifest_path
        if args.manifest_path is not None
        else (args.generated_config_dir / "jobs_manifest.csv")
    )
    write_manifest(bundles, manifest_path, args.mode, script_map)


if __name__ == "__main__":
    main()
