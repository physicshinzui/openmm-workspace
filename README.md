# openmm workspace

This repository provides a compact OpenMM workflow for energy minimisation, NVT, NPT, and production MD.
Production MD runs under NPT conditions because the barostat added for NPT equilibration remains active during production.
It is **config-first**: run parameters, file locations, reporting cadence, and restraint settings are managed in `config.yaml`.

## Repository layout

- `01_md.py` — thin CLI entry point for a single MD run
- `md_config.py` — YAML loading, validation, typed config objects, and path resolution
- `md_workflow.py` — OpenMM workflow implementation
- `batch_md.py` — thin CLI entry point for batch execution
- `batch_jobs.py` — batch job schema, config generation, and subprocess execution
- `config.yaml` — repository-default runnable sample config
- `jobs.yaml` — batch job definitions
- `environment.yml` — conda environment definition
- `1aki/1AKI.pdb` — sample input structure used by the root config

## Setup

```bash
conda env create -f environment.yml
conda activate openmm
```

## Single-run usage

```bash
python 01_md.py --config config.yaml
```

The root `config.yaml` points to `1aki/1AKI.pdb`, so it is a runnable sample config rather than a placeholder.

CLI flags are intentionally limited to run control:

- `--config` — choose the YAML config file
- `--restart` — resume production from a checkpoint
- `--checkpoint` — override `paths.checkpoint`
- `--until` — override the configured production horizon in nanoseconds

## Batch usage

```bash
python batch_md.py --jobs jobs.yaml --workers 1
```

Each job inherits a base config, overrides path-level fields such as `paths.pdb` or `paths.prmtop`/`paths.inpcrd` and `paths.run_id`, writes a generated config into `generated_configs/`, then launches `01_md.py` with that generated config.

### Config values shared by batch jobs

Jobs that select the same base config share every value in that config unless the
job overrides a `paths` value. If a job does not set `config`, it uses the file
selected by `batch_md.py --default-config`, which defaults to `config.yaml`.

The complete list of config values inherited from the base config is:

- `paths.input_format`
- `paths.pdb`, or `paths.prmtop` and `paths.inpcrd`
- `paths.output_root`
- `paths.run_id`
- `paths.topology`
- `paths.minimized`
- `paths.trajectory`
- `paths.log`
- `paths.checkpoint`
- `force_fields`
- `thermodynamics.temperature`
- `thermodynamics.pressure`
- `thermodynamics.friction_coefficient`
- `thermodynamics.step_size`
- `system.nonbonded_cutoff`
- `system.solvent_padding`
- `system.ionic_strength`
- `system.hydrogen_mass`
- `simulation.nvt_steps`
- `simulation.npt_steps`
- `simulation.production_steps`
- `reporting.dcd_interval`
- `reporting.stdout_interval`
- `reporting.log_interval`
- `restraints.force_constant`
- `restraints.selection`

The `force_fields` and `restraints` sections remain optional under the same rules
as a single-run config. Batch entries cannot directly override `force_fields`,
`thermodynamics`, `system`, `simulation`, `reporting`, or `restraints`. To use
different values for any of those sections, create another config file and select
it with the job's `config` field.

The following batch entry fields override values in the generated config:

| Batch field | Generated-config effect |
| --- | --- |
| `pdb` | Sets `paths.pdb` and removes `paths.prmtop` and `paths.inpcrd` |
| `prmtop` and `inpcrd` | Set `paths.prmtop` and `paths.inpcrd`, and remove `paths.pdb` |
| `output_root` | Sets `paths.output_root` |
| `run_id` | Sets `paths.run_id` |
| `paths` | Overrides any named key in the base config's `paths` mapping |
| `replicas` | Appends `_repNNN` to `paths.run_id` for each expanded replica |

When switching between PDB and Amber input modes, set `paths.input_format` to the
matching mode in the job's `paths` mapping, unless the selected base config leaves
it unset for auto-detection.

All other supported batch entry fields control job selection or launch behavior
without modifying the generated config:

| Batch field | Meaning |
| --- | --- |
| `name` | Job name used for generated files and log messages |
| `config` | Base config for this job |
| `until_ns` | Passes `--until` to `01_md.py` |
| `checkpoint` | Passes `--checkpoint` to `01_md.py`; does not change `paths.checkpoint` in the generated config |
| `restart` | Passes `--restart` to `01_md.py` |
| `extra_args` | Appends arguments to the `01_md.py` command |
| `env` | Adds environment variables for the launched process |

For Grid Engine systems, submit the whole batch as one array job with `qsub`:

```bash
python batch_md.py --jobs jobs.yaml --mode pbs --dry-run
python batch_md.py --jobs jobs.yaml --mode pbs
```

If a job sets `replicas: N`, the launcher expands it into `N` independent array tasks and appends a replica suffix to `paths.run_id` so the output directories do not collide.
Each PBS submission writes an immutable snapshot under `generated_configs/pbs_submissions/<submission-id>/`, so later submissions cannot overwrite the configs or task scripts of queued arrays.

Scheduler resources, queue, walltime, array concurrency, modules, and environment setup
are written directly in `scheduler/pbs/md_job.pbs.j2`. They are not configured in
`jobs.yaml`.

The default template uses Grid Engine syntax:

```bash
#$ -cwd
#$ -t 1-{job_count}
#$ -tc 4
```

`SGE_TASK_ID` identifies the array task. Edit the template directly for the target
cluster. Use `--pbs-template path/to/custom.pbs.j2` to select a different
hard-coded cluster template.

At minimum, each batch entry should define:

- `name`
- `pdb` or `prmtop`/`inpcrd`
- `run_id` if you want to control the output directory name
- `replicas` if you want multiple independent copies of the same input

## Post-processing a trajectory with MDTraj

To reimage a trajectory into the periodic box and center it after the run:

```bash
python center_trajectory.py \
  --trajectory traj.dcd \
  --topology minimized.pdb \
  --output traj_centered.dcd \
  --output-pdb traj_centered_first_frame.pdb
```

By default the script tries to use the MDTraj selection `protein` as the anchor molecule for imaging. If that selection matches no atoms, it falls back to MDTraj's default molecule-based imaging.

## `config.yaml` schema

### `paths`

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `input_format` | string | no | `pdb` or `amber`; if omitted, the loader auto-detects the mode |
| `pdb` | string | if using PDB input | Input PDB structure path |
| `prmtop` | string | if using Amber input | Input Amber topology (`tleap` `prmtop`) |
| `inpcrd` | string | if using Amber input | Input Amber coordinates (`tleap` `inpcrd`) |
| `output_root` | string | no | Root output directory, default `data/md_runs` |
| `run_id` | string | no | Name of the simulation subdirectory, default `default` |
| `topology` | string | yes | Output topology filename/path |
| `minimized` | string | yes | Output minimized-structure filename/path |
| `trajectory` | string | yes | Output trajectory filename/path |
| `log` | string | yes | Output CSV log filename/path |
| `checkpoint` | string | no | Output checkpoint filename/path, default `checkpoint.chk` |

**Path rules**

- Absolute paths are used as-is.
- Relative `paths.pdb`, `paths.prmtop`, `paths.inpcrd`, and `paths.output_root` are resolved from the current working directory.
- Relative output filenames like `topology.pdb` or `traj.dcd` are placed under `data/md_runs/<system_name>/<run_id>/output/`.
- Relative output paths with their own parent directory are resolved from the current working directory.

**Input modes**

- `paths.input_format: pdb`
  - requires `paths.pdb`
  - forbids `paths.prmtop`, `paths.inpcrd`
  - requires `force_fields`
- `paths.input_format: amber`
  - requires `paths.prmtop`, `paths.inpcrd`
  - forbids `paths.pdb`, `force_fields`
- If `paths.input_format` is omitted, the loader auto-detects the mode from the file paths.

### `force_fields`

- Type: non-empty list of strings for PDB input; optional for Amber input
- Meaning: OpenMM force-field XML files passed to `ForceField(...)` when starting from PDB

### `thermodynamics`

| Key | Type | Units | Required | Meaning |
| --- | --- | --- | --- | --- |
| `temperature` | number | K | yes | Simulation temperature |
| `pressure` | number | bar | yes | Target pressure |
| `friction_coefficient` | number | 1/ps | yes | Langevin friction coefficient |
| `step_size` | number | ps | yes | Integrator timestep |

### `system`

| Key | Type | Units | Required | Meaning |
| --- | --- | --- | --- | --- |
| `nonbonded_cutoff` | number | nm | yes | Nonbonded cutoff |
| `solvent_padding` | number | nm | PDB only | Solvent box padding |
| `ionic_strength` | number | molar | PDB only | Salt concentration |
| `hydrogen_mass` | number | amu | no | Hydrogen mass repartitioning target |

`solvent_padding` and `ionic_strength` are used only when building a new solvent box from PDB input. They are ignored for Amber `prmtop`/`inpcrd` input.

### `simulation`

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `nvt_steps` | integer | yes | NVT equilibration steps |
| `npt_steps` | integer | yes | NPT equilibration steps |
| `production_steps` | integer | yes | Production MD steps |

### `reporting`

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `dcd_interval` | integer | yes | Trajectory write interval |
| `stdout_interval` | integer | yes | Terminal reporter interval |
| `log_interval` | integer | yes | CSV log and checkpoint interval |

### `restraints`

Optional mapping.

| Key | Type | Required when section exists | Meaning |
| --- | --- | --- | --- |
| `force_constant` | number | yes | Positional restraint force constant in kJ/mol/nm² |
| `selection` | string | yes | MDAnalysis selection string used to choose restrained atoms |

If the `restraints` section is present, both fields must be set. A zero force constant is allowed and still validates. Restraints are applied during the configured NVT and NPT stages of a fresh run, then removed before production begins. Restart runs resume production without reapplying restraints.

## Outputs

Each run is stored as:

```text
data/md_runs/
  <system_name>/
    <run_id>/
      input/
      output/
```

Typical output files live under `output/`:

- `topology.pdb` — solvated topology after modeller preparation
- `minimized.pdb` — structure after minimisation
- `traj.dcd` — trajectory across all simulation stages
- `md_log.txt` — CSV state-data log
- `checkpoint.chk` — restart file

## Restarting

```bash
python 01_md.py --config config.yaml --restart
```

Restart resumes **production only**. It does not rerun minimisation, NVT, or NPT.

You can also:

```bash
python 01_md.py --config config.yaml --restart --checkpoint path/to/checkpoint.chk
python 01_md.py --config config.yaml --restart --until 1
```

- `--checkpoint` overrides `paths.checkpoint`
- `--until` is interpreted in **nanoseconds** and means the **target cumulative production length**

For restart to work, you must use the **same config lineage** as the original run:

- the config must resolve to the same `paths.topology`
- the config must resolve to the same `paths.checkpoint`
- the saved `topology.pdb` and checkpoint file must both still exist

For example, if a run was started with `generated_configs/smoke_test.yaml`, restart it with that same config:

```bash
python 01_md.py --config generated_configs/smoke_test.yaml --restart --until 1
```

Using a different config file is fine only if it still points to the same saved topology and checkpoint paths.
