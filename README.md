# openmm workspace

This repository contains a compact yet extensible OpenMM workflow that covers preparation, replica production, and Markov state model (MSM) analysis. Configuration lives in YAML, simulations are reproducible through metadata sidecars, and helper scripts streamline running large batches on PBS-like schedulers and post-processing the resulting trajectories.

## Repository Layout
- `01_md.py` — main MD driver. Supports replica-aware paths, random seeding, restart logic, and emits per-run metadata JSON.
- `batch_md.py` — orchestrates multiple jobs driven by `jobs.yaml`. Can run locally or generate PBS scripts (with optional submission) and writes a CSV job manifest.
- `config.yaml` — central configuration for force fields, simulation stages, replica layout, and scheduler defaults.
- `analysis/collect_runs.py` — scans metadata JSON files and produces a manifest of completed replicas.
- `analysis/featurize.py` — loads the run manifest, computes MSM-friendly features (backbone dihedrals or CA distances), and records a feature manifest.
- `analysis/build_msm.py` — consumes the feature manifest and builds a deeptime MSM (TICA → clustering → MSM) with optional embeddings.
- `utils/monitor_basic_quantity.py` / `utils/trjconv.py` — legacy analysis helpers for quick trajectory inspection.
- `scheduler/pbs/md_job.pbs.j2` — PBS template rendered by `batch_md.py --mode pbs`.

## Environment Setup
1. Install OpenMM (CPU or GPU build).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   For MSM analysis you additionally need `mdtraj` and `deeptime` (see `environment.yml` for a conda example).

## Single-Run Execution
```
python 01_md.py --config config.yaml
```

Key CLI options:
- `--replica-id` — override the replica index (defaults to `REPLICA_ID`, `PBS_ARRAY_INDEX`, or 0).
- `--seed` — explicit random seed (otherwise derived from `replicas.seed_start`).
- `--stage production` — continue from checkpoint only (requires `--restart`).
- `--until <ns>` — cap production time in nanoseconds.

Outputs are written beneath `paths.output_root/<pdb-stem>/` with replica-specific subdirectories such as `simulations/<run_id>/replica_000/`. Each run produces `metadata.json` capturing the CLI parameters, seeds, step counts, and paths to artefacts (trajectory, log, checkpoint, topology). These metadata files drive the downstream collection scripts.

## Replica & Scheduler Configuration (`config.yaml`)
- `paths` — input/output files (relative paths resolve under `paths.output_root`).
- `replicas` — controls replica bookkeeping; adjust `count`, `directory_pattern`, seeds, and metadata filename.
- `scheduler` — default PBS settings (`template`, `queue`, `walltime`, resource block, module/pre-command hooks, and qsub flags). `batch_md.py` merges job-specific overrides with these defaults.
- `force_fields`, `thermodynamics`, `system`, `simulation`, `reporting`, `restraints` — standard OpenMM options (see inline comments).

## Batch Execution & PBS Submission
1. List desired systems in `jobs.yaml`. Each job can override `pdb`, `run_id`, `replicas`, scheduler settings (`scheduler:` block), extra CLI flags, and environment variables.
2. Generate configs and PBS scripts:
   ```bash
   python batch_md.py \
     --jobs jobs.yaml \
     --mode pbs \
     --dry-run \
     --template scheduler/pbs/md_job.pbs.j2
   ```
   Remove `--dry-run` to create scripts under `scheduler/pbs/generated/` and optional logs under `scheduler/pbs/logs/`.
3. Submit automatically with `--submit` (uses `scheduler.submit_command` and `submit_flags`), or invoke `qsub` manually.
4. Inspect `generated_configs/jobs_manifest.csv` for a record of replica commands, configuration files, and PBS script paths.

### Local Execution
```
python batch_md.py --jobs jobs.yaml --mode local --workers 2
```
Launches replicas directly via `subprocess` (respecting environment overrides).

## Post-Processing Pipeline
1. **Collect metadata**
   ```bash
   python analysis/collect_runs.py \
     --config config.yaml \
     --output data/manifests/runs_manifest.csv \
     --json-output data/manifests/runs_manifest.json \
     --verbose
   ```
   Filters metadata files, verifies trajectories, and summarises production lengths.

2. **Featurise trajectories**
   ```bash
   python analysis/featurize.py \
     --manifest data/manifests/runs_manifest.csv \
     --output-dir data/features \
     --feature-set backbone-dihedrals \
     --stride 10 \
     --verbose
   ```
   Generates per-replica `.npz` files containing feature matrices, metadata, and frame timing; emits `data/features/feature_manifest.csv`.

3. **Build MSM with deeptime**
   ```bash
   python analysis/build_msm.py \
     --features-manifest data/features/feature_manifest.csv \
     --output-dir data/msm/protein \
     --lag-time 100 \
     --tica-dim 8 \
     --cluster-k 300 \
     --reversible \
     --save-embeddings \
     --verbose
   ```
   Stores fitted models (`models/*.pkl`), implied timescales (`timescales.csv`), optional trajectory embeddings, and a `summary.json` with provenance.

## Legacy Analysis Helpers
`utils/monitor_basic_quantity.py` still provides quick-look plots (RMSD, RMSF, radius of gyration, Cp) from `md_log.txt`. Invoke as:
```bash
python utils/monitor_basic_quantity.py --selection "backbone and not name H*"
```

## Tips
- Replica metadata drives every downstream step—keep the generated `metadata.json` files alongside trajectories.
- Adjust `scheduler.environment.modules` and `pre_commands` to match your cluster’s module system or conda bootstrap.
- Prefer `--dry-run` when regenerating PBS scripts to verify template substitutions without touching the queue.
- For heterogeneous frame spacing ensure each replica uses the same `dcd_interval` and stride; `build_msm.py` currently assumes a consistent frame timestep across trajectories.
