# openmm workspace

This repository provides a compact OpenMM workflow for energy minimisation, NVT, NPT, and production MD.
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

Each job inherits a base config, overrides path-level fields such as `paths.pdb` and `paths.run_id`, writes a generated config into `generated_configs/`, then launches `01_md.py` with that generated config.

## `config.yaml` schema

### `paths`

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `pdb` | string | yes | Input PDB structure path |
| `output_root` | string | no | Root output directory, default `data/md_runs` |
| `run_id` | string | no | Name of the simulation subdirectory, default `default` |
| `topology` | string | yes | Output topology filename/path |
| `minimized` | string | yes | Output minimized-structure filename/path |
| `trajectory` | string | yes | Output trajectory filename/path |
| `log` | string | yes | Output CSV log filename/path |
| `checkpoint` | string | no | Output checkpoint filename/path, default `checkpoint.chk` |

**Path rules**

- Absolute paths are used as-is.
- Relative `paths.pdb` and `paths.output_root` are resolved from the current working directory.
- Relative output filenames like `topology.pdb` or `traj.dcd` are placed under the workflow directories derived from `output_root` and `run_id`.
- Relative output paths with their own parent directory are resolved from the current working directory.

### `force_fields`

- Type: non-empty list of strings
- Meaning: OpenMM force-field XML files passed to `ForceField(...)`

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
| `solvent_padding` | number | nm | yes | Solvent box padding |
| `ionic_strength` | number | molar | yes | Salt concentration |
| `hydrogen_mass` | number | amu | no | Hydrogen mass repartitioning target |

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
