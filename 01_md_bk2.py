from __future__ import annotations

from pathlib import Path
from sys import stdout

import openmm
from openmm import MonteCarloBarostat, LangevinMiddleIntegrator, unit
from openmm.app import (
    DCDReporter,
    ForceField,
    HBonds,
    Modeller,
    PME,
    PDBFile,
    Simulation,
    StateDataReporter,
)


PDB_PATH = Path("1AKI.pdb")
TOPOLOGY_PATH = Path("top.pdb")
TRAJECTORY_PATH = Path("traj.dcd")
LOG_PATH = Path("md_log.txt")

FORCE_FIELD_FILES = ("amber19/protein.ff19SB.xml", "amber19/tip3pfb.xml")

TEMPERATURE = 300 * unit.kelvin
PRESSURE = 1 * unit.bar
FRICTION_COEFFICIENT = 1 / unit.picosecond
STEP_SIZE = 0.004 * unit.picoseconds

NONBONDED_CUTOFF = 1.0 * unit.nanometer
SOLVENT_PADDING = 1.0 * unit.nanometer
IONIC_STRENGTH = 0.15 * unit.molar

NVT_STEPS = 10_000
NPT_STEPS = 10_000

DCD_INTERVAL = 1_000
STDOUT_INTERVAL = 1_000
LOG_INTERVAL = 100


def build_modeller(forcefield: ForceField) -> Modeller:
    pdb = PDBFile(str(PDB_PATH))
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield, pH=7.0)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        boxShape="cube",
        padding=SOLVENT_PADDING,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=IONIC_STRENGTH,
    )
    return modeller


def build_system(modeller: Modeller, forcefield: ForceField) -> openmm.System:
    return forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=NONBONDED_CUTOFF,
        constraints=HBonds,
        removeCMMotion=True,
    )


def build_simulation(
    modeller: Modeller, system: openmm.System
) -> Simulation:
    integrator = LangevinMiddleIntegrator(
        TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE
    )
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    return simulation


def write_topology(modeller: Modeller) -> None:
    with TOPOLOGY_PATH.open("w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)


def attach_reporters(simulation: Simulation) -> None:
    simulation.reporters.extend(
        [
            DCDReporter(
                str(TRAJECTORY_PATH),
                DCD_INTERVAL,
                enforcePeriodicBox=True,
            ),
            StateDataReporter(
                stdout,
                STDOUT_INTERVAL,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                speed=True,
            ),
            StateDataReporter(
                str(LOG_PATH),
                LOG_INTERVAL,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                speed=True,
            ),
        ]
    )


def run_nvt(simulation: Simulation, steps: int) -> None:
    print("Running NVT")
    simulation.step(steps)


def run_npt(
    simulation: Simulation, system: openmm.System, steps: int
) -> None:
    system.addForce(MonteCarloBarostat(PRESSURE, TEMPERATURE))
    simulation.context.reinitialize(preserveState=True)
    print("Running NPT")
    simulation.step(steps)


def main() -> None:
    forcefield = ForceField(*FORCE_FIELD_FILES)
    modeller = build_modeller(forcefield)
    write_topology(modeller)

    system = build_system(modeller, forcefield)
    simulation = build_simulation(modeller, system)
    attach_reporters(simulation)

    print("Minimizing energy")
    simulation.minimizeEnergy()
    run_nvt(simulation, NVT_STEPS)
    run_npt(simulation, system, NPT_STEPS)


if __name__ == "__main__":
    main()
