from __future__ import annotations

import argparse
import sys
from pathlib import Path

from batch_jobs import (
    DEFAULT_GENERATED_DIR,
    DEFAULT_JOBS_PATH,
    DEFAULT_MD_SCRIPT,
    run_batch_jobs,
)
from md_config import DEFAULT_CONFIG_PATH


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


def main() -> None:
    args = parse_args()
    run_batch_jobs(
        jobs_path=args.jobs,
        default_config_path=args.default_config,
        generated_config_dir=args.generated_config_dir,
        md_script=args.md_script,
        python_executable=args.python,
        workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
