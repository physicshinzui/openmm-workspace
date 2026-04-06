# openmm workspace

<<<<<<< HEAD
This repository contains a compact yet extensible OpenMM workflow that covers preparation, replica production, and Markov state model (MSM) analysis. Configuration lives in YAML, simulations are reproducible through metadata sidecars, and helper scripts streamline running large batches on PBS-like schedulers and post-processing the resulting trajectories.

## Repository Layout
- `01_md.py` — main MD driver. Supports replica-aware paths, random seeding, restart logic, and emits per-run metadata JSON.
- `batch_md.py` — orchestrates multiple jobs driven by `jobs.yaml`. Can run locally or generate PBS scripts (with optional submission) and writes a CSV job manifest.
- `config.yaml` — central configuration for force fields, simulation stages, replica layout, and scheduler defaults.
- `analysis/collect_runs.py` — scans metadata JSON files and produces a manifest of completed replicas.
- `analysis/featurize.py` — loads the run manifest, computes MSM-friendly features (backbone dihedrals or CA distances), and records a feature manifest.
- `analysis/build_msm.py` — consumes the feature manifest and builds a deeptime MSM (TICA → clustering → MSM) with optional embeddings.
- `analysis/run_pipeline.py` — convenience wrapper that chains the above three steps (collect → featurise → MSM).
- `analysis/plot_msm.py` — optional plotting tool (implied timescales, TICA scatter, CK test) for MSM results.
- `analysis/inspect_msm.py` — CLI utility to print stationary distributions, MFPTs, and optional PCCA+ memberships.
- `utils/monitor_basic_quantity.py` / `utils/trjconv.py` — legacy analysis helpers for quick trajectory inspection.
- `scheduler/pbs/md_job.pbs.j2` — PBS template rendered by `batch_md.py --mode pbs`.

## Environment Setup
1. Install OpenMM (CPU or GPU build).
2. Install Python dependencies:
=======
This directory provides a compact OpenMM workflow for energy minimisation, NVT, NPT, and production MD.
The repository is now **config-first**: simulation parameters, file locations, reporting intervals, and restraint settings are managed in YAML.

## Contents
- `01_md.py` — thin CLI entry point for a single MD run.
- `md_config.py` — shared config models, YAML loading, validation, and path resolution.
- `md_workflow.py` — OpenMM workflow implementation for minimisation, equilibration, production, and restart handling.
- `batch_md.py` — thin CLI entry point for batch execution.
- `batch_jobs.py` — batch job schema, config generation, and subprocess execution logic.
- `config.yaml` — central configuration for inputs, outputs, simulation phases, and reporting cadence.
- `jobs.yaml` — list of structures and run IDs for batch execution.
- `environment.yml` — conda environment specification for the MD workflow.
- `1aki/1AKI.pdb` — sample starting structure.

## Setup
1. Create the conda environment:
>>>>>>> bd32642 (Refactor MD workflow into config-first modules and remove analysis code)
   ```bash
   conda env create -f environment.yml
   conda activate openmm
   ```
<<<<<<< HEAD
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
=======
2. Set `paths.pdb` in `config.yaml`, or use `jobs.yaml` to inject per-job input structures.

## Running a single simulation
```bash
python 01_md.py --config config.yaml
```

CLI flags are intentionally limited to run control:
- `--config` — choose the YAML config file.
- `--restart` — resume production from a checkpoint.
- `--checkpoint` — override the configured checkpoint path.
- `--until` — override the configured production horizon in nanoseconds.

Pipeline summary:
1. Read the input PDB, remove waters, add hydrogens, solvate, and ionise.
2. Minimise energy and write the structure to `paths.minimized`.
3. Run NVT and NPT while positional restraints are active for the configured selection.
4. Remove restraints and continue with the production phase.
5. Emit `traj.dcd` and `md_log.txt` at the configured intervals.
>>>>>>> bd32642 (Refactor MD workflow into config-first modules and remove analysis code)

Outputs are written beneath `paths.output_root/<pdb-stem>/` with replica-specific subdirectories such as `simulations/<run_id>/replica_000/`. Each run produces `metadata.json` capturing the CLI parameters, seeds, step counts, and paths to artefacts (trajectory, log, checkpoint, topology). These metadata files drive the downstream collection scripts.

<<<<<<< HEAD
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

### One-shot Pipeline
The helper script below strings the three stages together; customise arguments as needed:
```bash
python analysis/run_pipeline.py \
  --config config.yaml \
  --run-manifest data/manifests/protein_runs.csv \
  --feature-dir data/features/protein \
  --msm-dir data/msm/protein \
  --feature-set backbone-dihedrals \
  --lag-time 50 \
  --stride 10 \
  --reversible \
  --save-embeddings \
  --verbose
```

### Visualization
Use the plotting helper to generate diagnostics once the MSM has been built:
```bash
python analysis/plot_msm.py \
  --msm-dir data/msm/protein \
  --embeddings-manifest data/msm/protein/embedding_manifest.csv \
  --pair 0 1 \
  --ck-lags 1 2 4 \
  --verbose
```
This writes PNGs (timescales, TICA scatter, optional CK test) into `data/msm/protein/figures/`.

### Inspect MSM Numerically
To print stationary populations, MFPT matrices, and optionally PCCA+ memberships (with CSV export), run:
```bash
python analysis/inspect_msm.py \
  --msm-dir data/msm/protein \
  --macro-states 3 \
  --output-csv
```

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
=======
- `paths`
  - `pdb` / `output_root` / `run_id` / `topology` / `minimized` / `trajectory` / `log` / `checkpoint`
- `force_fields`
- `thermodynamics`
  - temperature, pressure, friction coefficient, and integrator step size
- `system`
  - non-bonded cutoff, solvent padding, ionic strength, and optional `hydrogen_mass`
- `simulation`
  - step counts for NVT, NPT, and production
- `reporting`
  - DCD, stdout, and log intervals
- `restraints`
  - positional restraint force constant and MDAnalysis selection string

## Outputs
- `topology.pdb` — solvated topology written after the modeller stage
- `minimized.pdb` — structure after minimisation
- `traj.dcd` — trajectory covering all simulation stages
- `md_log.txt` — CSV log containing step, time (ps), energies, temperature, volume, density, and performance (ns/day)
- `checkpoint.chk` — latest restart state written by `CheckpointReporter`

## Batch execution
Batch runs are driven by `jobs.yaml`.

```bash
python batch_md.py --jobs jobs.yaml --workers 1
```

For each job, the batch runner:
1. Loads a base `config.yaml`.
2. Overrides `paths.pdb`, `paths.run_id`, and any job-specific path settings.
3. Writes a generated config into `generated_configs/`.
4. Launches `01_md.py --config <generated-config>` in a subprocess.

## Restarting from a checkpoint
- Configure the checkpoint path via `paths.checkpoint` (default `checkpoint.chk`).
- Resume production with:
  ```bash
  python 01_md.py --config config.yaml --restart
  ```
- Override the checkpoint path on the command line with `--checkpoint custom.chk`.
- Target a specific production horizon in nanoseconds with `--until`.
- Keep `topology.pdb`; it is required for restart because it stores the solvated topology.
>>>>>>> bd32642 (Refactor MD workflow into config-first modules and remove analysis code)
