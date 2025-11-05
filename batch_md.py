from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


@dataclass(frozen=True)
class PreparedJob:
    index: int
    name: str
    command: List[str]
    env: Dict[str, str]
    config_path: Path


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
        "--workers",
        type=int,
        default=1,
        help="Number of jobs to run concurrently. Use 1 for sequential execution.",
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
) -> List[str]:
    command = [python_exe, str(md_script), "--config", str(config_path)]
    checkpoint = job.get("checkpoint")
    if checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])
    if job.get("restart"):
        command.append("--restart")
    until_ns = job.get("until_ns")
    if until_ns is not None:
        command.extend(["--until", str(until_ns)])
    for arg in job.get("extra_args", []):
        command.append(str(arg))
    return command


def prepare_jobs(
    args: argparse.Namespace,
    jobs_data: Iterable[Dict[str, Any]],
) -> List[PreparedJob]:
    base_config_cache: Dict[Path, Dict[str, Any]] = {}
    prepared: List[PreparedJob] = []
    for index, job in enumerate(jobs_data, start=1):
        job_name = derive_job_name(index, job)
        config_path = Path(job.get("config", args.default_config))
        base_config = load_base_config(config_path, base_config_cache)
        merged_config = prepare_config(base_config, job, index)
        written_config = write_config(job_name, args.generated_config_dir, merged_config)

        command = build_command(args.md_script, args.python, written_config, job)
        env = os.environ.copy()
        for key, value in job.get("env", {}).items():
            env[str(key)] = str(value)

        prepared.append(
            PreparedJob(
                index=index,
                name=job_name,
                command=command,
                env=env,
                config_path=written_config,
            )
        )
    return prepared


def execute_job(job: PreparedJob, dry_run: bool) -> int:
    cmd_str = " ".join(job.command)
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


def run_jobs(prepared: List[PreparedJob], workers: int, dry_run: bool) -> None:
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


def main() -> None:
    args = parse_args()
    jobs_data = ensure_list(load_yaml(args.jobs), args.jobs)
    prepared = prepare_jobs(args, jobs_data)
    run_jobs(prepared, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
