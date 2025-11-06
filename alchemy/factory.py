from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import openmm
from openmmtools import alchemy

from core import HREXAlchemicalRegionConfig, select_restraint_atoms


@dataclass(frozen=True)
class AlchemicalPreparation:
    """Container for alchemical system components."""

    system: openmm.System
    region: alchemy.AlchemicalRegion
    atom_indices: Tuple[int, ...]
    factory: alchemy.AbsoluteAlchemicalFactory


def resolve_alchemical_atom_indices(
    topology_path: Path, selection: str
) -> Tuple[int, ...]:
    """Select atom indices to alchemically modify."""
    indices = select_restraint_atoms(topology_path, selection)
    return tuple(indices)


def build_alchemical_region(
    atom_indices: Sequence[int], region_cfg: HREXAlchemicalRegionConfig
) -> alchemy.AlchemicalRegion:
    """Construct an openmmtools AlchemicalRegion from configuration."""
    kwargs: dict[str, object] = {
        "alchemical_atoms": list(atom_indices),
        "annihilate_electrostatics": region_cfg.annihilate_electrostatics,
        "annihilate_sterics": region_cfg.annihilate_sterics,
    }
    if region_cfg.softcore_alpha is not None:
        kwargs["softcore_alpha"] = region_cfg.softcore_alpha
    if region_cfg.softcore_beta is not None:
        kwargs["softcore_beta"] = region_cfg.softcore_beta
    if region_cfg.softcore_a is not None:
        kwargs["softcore_a"] = region_cfg.softcore_a
    if region_cfg.softcore_b is not None:
        kwargs["softcore_b"] = region_cfg.softcore_b
    return alchemy.AlchemicalRegion(**kwargs)


def prepare_alchemical_system(
    base_system: openmm.System,
    topology_path: Path,
    region_cfg: HREXAlchemicalRegionConfig,
    *,
    selection_override: Optional[str] = None,
    factory_kwargs: Optional[Mapping[str, object]] = None,
) -> AlchemicalPreparation:
    """Create an alchemically-modified system and associated metadata."""
    selection = selection_override or region_cfg.selection
    atom_indices = resolve_alchemical_atom_indices(topology_path, selection)
    region = build_alchemical_region(atom_indices, region_cfg)
    factory = alchemy.AbsoluteAlchemicalFactory(
        **(dict(factory_kwargs) if factory_kwargs is not None else {})
    )
    alchemical_system = factory.create_alchemical_system(base_system, region)
    return AlchemicalPreparation(
        system=alchemical_system,
        region=region,
        atom_indices=atom_indices,
        factory=factory,
    )
