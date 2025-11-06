from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from openmmtools import mcmc, multistate, states

from core import HREXConfig


def _build_mcmc_move(hrex_config: HREXConfig) -> mcmc.MCMCMove:
    splitting = hrex_config.mcmc.splitting.strip()
    if splitting:
        move = mcmc.LangevinSplittingDynamicsMove(
            timestep=hrex_config.mcmc.timestep,
            collision_rate=hrex_config.mcmc.collision_rate,
            n_steps=hrex_config.mcmc.n_steps_per_iteration,
            splitting=splitting,
        )
    else:
        move = mcmc.LangevinDynamicsMove(
            timestep=hrex_config.mcmc.timestep,
            collision_rate=hrex_config.mcmc.collision_rate,
            n_steps=hrex_config.mcmc.n_steps_per_iteration,
        )
    return move


def build_replica_exchange_sampler(hrex_config: HREXConfig) -> multistate.ReplicaExchangeSampler:
    sampler = multistate.ReplicaExchangeSampler(
        mcmc_moves=_build_mcmc_move(hrex_config),
        number_of_iterations=hrex_config.iterations.production_iterations,
    )
    return sampler


def create_reporter(hrex_config: HREXConfig, mode: str = "w") -> multistate.MultiStateReporter:
    storage_path = str(hrex_config.paths.storage_path)
    reporter = multistate.MultiStateReporter(
        storage=storage_path,
        open_mode=mode,
        checkpoint_interval=hrex_config.iterations.checkpoint_interval,
        checkpoint_storage=hrex_config.paths.checkpoint_filename,
        analysis_particle_indices=hrex_config.reporting.analysis_particle_indices,
        position_interval=hrex_config.reporting.position_interval,
        velocity_interval=hrex_config.reporting.velocity_interval,
    )
    return reporter


@dataclass
class HREXRunConfig:
    minimize: bool = True
    equilibration_iterations: Optional[int] = None
    production_iterations: Optional[int] = None

    @classmethod
    def from_hrex(cls, hrex_config: HREXConfig) -> "HREXRunConfig":
        return cls(
            minimize=True,
            equilibration_iterations=hrex_config.iterations.equilibration_iterations,
            production_iterations=hrex_config.iterations.production_iterations,
        )


class ReplicaExchangeController:
    """Thin wrapper that orchestrates ReplicaExchangeSampler runs."""

    def __init__(
        self,
        sampler: multistate.ReplicaExchangeSampler,
        reporter: multistate.MultiStateReporter,
        hrex_config: HREXConfig,
    ) -> None:
        self.sampler = sampler
        self.reporter = reporter
        self.hrex_config = hrex_config

    def create(
        self,
        thermodynamic_states: Sequence[states.CompoundThermodynamicState],
        sampler_state: states.SamplerState,
        state_indices: Optional[Sequence[int]] = None,
    ) -> None:
        kwargs = {}
        if state_indices is not None:
            kwargs["initial_thermodynamic_states"] = state_indices
        self.sampler.create(
            thermodynamic_states=thermodynamic_states,
            sampler_states=sampler_state,
            storage=self.reporter,
            **kwargs,
        )

    def run(self, run_config: Optional[HREXRunConfig] = None) -> None:
        config = run_config or HREXRunConfig.from_hrex(self.hrex_config)
        if config.minimize:
            self.sampler.minimize()
        if config.equilibration_iterations:
            self.sampler.equilibrate(config.equilibration_iterations)
        if config.production_iterations:
            self.sampler.extend(config.production_iterations)

    def resume(self, production_iterations: Optional[int] = None) -> None:
        iterations = (
            production_iterations
            if production_iterations is not None
            else self.hrex_config.iterations.production_iterations
        )
        if iterations:
            self.sampler.extend(iterations)

    def close(self) -> None:
        self.reporter.close()


def resume_from_storage(storage_path: Path) -> tuple[
    multistate.ReplicaExchangeSampler, multistate.MultiStateReporter
]:
    reporter = multistate.MultiStateReporter(str(storage_path), open_mode="r+")
    sampler = multistate.ReplicaExchangeSampler.from_storage(reporter)
    return sampler, reporter
