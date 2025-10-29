from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import openmm
from openmm import app
from openmm import unit
from io import StringIO
from typing import Optional

import MDAnalysis as mda
from MDAnalysis.transformations import wrap, unwrap

def create_position_restraint_force(topology: app.Topology, 
                                    reference_positions: unit.Quantity, 
                                    force_constant: unit.Quantity, 
                                    selection_query: str = "not element H") -> openmm.CustomExternalForce:
    """
    Creates a CustomExternalForce object to apply harmonic position restraints 
    to a specified set of atoms.

    The potential energy function is:
    E = 0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)

    Args:
        topology: openmm.app.Topology
            The topology of the system.
        reference_positions: unit.Quantity
            The reference coordinates (e.g., from a PDB file) to restrain the atoms to.
        force_constant: unit.Quantity
            The spring constant (k) for the restraint. 
            (e.g., 1000.0 * unit.kilojoules_per_mole / (unit.nanometer**2))
        selection_query: str
            A query string to select atoms for restraint.
            Supported queries:
            - "not element H" (default): Restrain heavy atoms (non-hydrogen).
            - "name CA": Restrain C-alpha atoms.
            - "all": Restrain all atoms.

    Returns:
        openmm.CustomExternalForce
            The configured CustomExternalForce object, ready to be added to the System.
    """
    
    energy_function = "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    restraint_force = openmm.CustomExternalForce(energy_function)

    # Define the per-particle parameters required by the energy function
    restraint_force.addPerParticleParameter("k")
    restraint_force.addPerParticleParameter("x0")
    restraint_force.addPerParticleParameter("y0")
    restraint_force.addPerParticleParameter("z0")

    # Select atoms based on the query
    if selection_query == "not element H":
        selector = lambda atom: atom.element.symbol != 'H'
    elif selection_query == "name CA":
        selector = lambda atom: atom.name == 'CA'
    elif selection_query == "all":
        selector = lambda atom: True
    else:
        raise ValueError(f"Unsupported selection query: {selection_query}")

    # Loop over atoms and add them to the force if selected
    for atom in topology.atoms():
        if selector(atom):
            # Get the reference position for this atom
            ref_pos = reference_positions[atom.index]
            
            # Set the parameters (k, x0, y0, z0)
            # We must use .value_in_unit() to pass unitless floats to addParticle,
            # ensuring they are in the standard units (kJ/mol, nm).
            parameters = [
                force_constant.value_in_unit(unit.kilojoules_per_mole / (unit.nanometer**2)),
                ref_pos[0].value_in_unit(unit.nanometer),
                ref_pos[1].value_in_unit(unit.nanometer),
                ref_pos[2].value_in_unit(unit.nanometer)
            ]
            
            # Add the particle (atom) and its parameters to the force
            restraint_force.addParticle(atom.index, parameters)
            
    return restraint_force



pdb = PDBFile("1AKI.pdb")

# Specify the forcefield
forcefield = ForceField('amber19/protein.ff19SB.xml', 'amber19/tip3pfb.xml')

modeller = Modeller(pdb.topology, pdb.positions)
modeller.deleteWater()
residues = modeller.addHydrogens(
             forcefield,
             pH=7.0,
             )

#modeller.addSolvent(forcefield, padding=1.0*nanometer)

modeller.addSolvent(
    forcefield,
    model='tip3p',
    boxShape='cube',
    padding=1.0 * nanometer,
    positiveIon='Na+',
    negativeIon='Cl-',
    ionicStrength=0.15 * molar, 
#    neutralize=True,
)

# Create a topology file
PDBFile.writeFile(modeller.topology, 
                  modeller.positions, 
                  open('top.pdb', 'w'))

system = forcefield.createSystem(modeller.topology, 
                                 nonbondedMethod=PME, 
                                 nonbondedCutoff=1.0*nanometer, 
                                 constraints=HBonds,
                                 removeCMMotion=True, # Default
                                 )

integrator = LangevinMiddleIntegrator(300*kelvin, 
                                      1/picosecond, 
                                      0.004*picoseconds
                                      )
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

print("Minimizing energy")
simulation.minimizeEnergy()

simulation.reporters.append(DCDReporter(
    'traj.dcd', 
    1000,
    enforcePeriodicBox=True
    )
)
simulation.reporters.append(
        StateDataReporter(
            stdout, 1000, 
            step=True,
            potentialEnergy=True, 
            temperature=True, 
            volume=True,
            speed=True
            )
        )
simulation.reporters.append(
        StateDataReporter(
            "md_log.txt", 100, 
            step=True,
            potentialEnergy=True, 
            temperature=True, 
            volume=True,
            speed=True
            )
        )

print("Running NVT")
simulation.step(10000)

system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
simulation.context.reinitialize(preserveState=True)

print("Running NPT")
simulation.step(10000)
