from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from openmmtools.multistate import MultiStateReporter
from openmmtools.multistate.multistateanalyzer import MultiStateSamplerAnalyzer


@dataclass(frozen=True)
class AcceptanceStatistics:
    overall: float
    matrix: np.ndarray
    accepted: np.ndarray
    proposed: np.ndarray


def load_multistate_reporter(storage_path: Path, mode: str = "r") -> MultiStateReporter:
    """Open a MultiStateReporter for the provided storage file."""
    return MultiStateReporter(str(storage_path), open_mode=mode)


def analyse_acceptance(reporter: MultiStateReporter) -> AcceptanceStatistics:
    """Compute acceptance statistics aggregated over all recorded iterations."""
    accepted, proposed = reporter.read_mixing_statistics()
    total_accepted = accepted.sum(axis=0)
    total_proposed = proposed.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(total_proposed > 0, total_accepted / total_proposed, 0.0)
    overall = (
        float(total_accepted.sum()) / float(total_proposed.sum())
        if total_proposed.sum() > 0
        else 0.0
    )
    return AcceptanceStatistics(
        overall=overall,
        matrix=matrix,
        accepted=total_accepted,
        proposed=total_proposed,
    )


def extract_free_energy_differences(
    reporter: MultiStateReporter,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return MBAR free energy differences Δf_ij and their uncertainties."""
    analyzer = MultiStateSamplerAnalyzer(reporter)
    delta_f_ij, d_delta_f_ij = analyzer.get_free_energy()
    return np.asarray(delta_f_ij), np.asarray(d_delta_f_ij)
