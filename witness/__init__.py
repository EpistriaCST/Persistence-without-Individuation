"""Witness system: two-oscillator Langevin substrate from §3 of the paper."""

from witness.dynamics import (
    drift_matrix,
    diffusion_matrix,
    transformation_T,
    simulate,
)
from witness.partitions import (
    Partition,
    P1_community,
    P2_normal_modes,
    apply_transformation,
)
from witness.stationary import (
    stationary_covariance_analytical,
    stationary_covariance_numerical,
    lagged_covariance,
)

__all__ = [
    "drift_matrix",
    "diffusion_matrix",
    "transformation_T",
    "simulate",
    "Partition",
    "P1_community",
    "P2_normal_modes",
    "apply_transformation",
    "stationary_covariance_analytical",
    "stationary_covariance_numerical",
    "lagged_covariance",
]
