from __future__ import annotations

import copy
from typing import List

import openmm
from openmmtools import alchemy, states

from alchemy import AlchemicalPreparation
from core import HREXConfig, SimulationConfig


def create_thermodynamic_state(
    alchemical_prep: AlchemicalPreparation,
    sim_config: SimulationConfig,
    hrex_config: HREXConfig,
) -> states.CompoundThermodynamicState:
    """Create the reference CompoundThermodynamicState for HREX."""
    system = alchemical_prep.system
    pressure = hrex_config.protocol.pressure
    barostat_frequency = hrex_config.protocol.barostat_frequency
    if pressure is not None:
        if barostat_frequency is None:
            raise ValueError(
                "HREX protocol specifies a pressure but no barostat_frequency."
            )
        has_barostat = any(
            isinstance(system.getForce(index), openmm.MonteCarloBarostat)
            for index in range(system.getNumForces())
        )
        if not has_barostat:
            barostat = openmm.MonteCarloBarostat(
                pressure,
                sim_config.temperature,
                barostat_frequency,
            )
            system.addForce(barostat)

    thermodynamic_state = states.ThermodynamicState(
        system=system,
        temperature=sim_config.temperature,
        pressure=pressure,
    )

    alchemical_state = alchemy.AlchemicalState.from_system(alchemical_prep.system)
    return states.CompoundThermodynamicState(
        thermodynamic_state=thermodynamic_state,
        composable_states=[alchemical_state],
    )


def build_compound_states(
    alchemical_prep: AlchemicalPreparation,
    sim_config: SimulationConfig,
    hrex_config: HREXConfig,
) -> List[states.CompoundThermodynamicState]:
    """Generate CompoundThermodynamicState objects for all replicas."""
    reference_state = create_thermodynamic_state(alchemical_prep, sim_config, hrex_config)
    lambda_schedule = hrex_config.protocol.lambda_schedule
    temperature_schedule = hrex_config.protocol.temperature_schedule

    compound_states: List[states.CompoundThermodynamicState] = []
    for index, lambda_value in enumerate(lambda_schedule):
        state = copy.deepcopy(reference_state)
        state.lambda_sterics = lambda_value
        state.lambda_electrostatics = lambda_value
        if temperature_schedule is not None:
            state.temperature = temperature_schedule[index]
        compound_states.append(state)

    return compound_states


def build_sampler_state(
    *,
    positions,
    box_vectors=None,
) -> states.SamplerState:
    """Create an initial SamplerState with provided positions and box vectors."""
    sampler_state = states.SamplerState(positions=positions, box_vectors=box_vectors)
    return sampler_state
