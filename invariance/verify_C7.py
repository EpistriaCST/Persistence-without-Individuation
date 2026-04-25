"""Verify Appendix C.7: Kuramoto order parameter is structurally near-invariant.

Like PLV, the Kuramoto r is not exactly invariant under T in G. The structural
near-invariance is at order O((k/omega_0^2)^2): the time-averaged r in the
original basis and in the normal-mode basis differ by a quantity that scales
quadratically with the dimensionless coupling.

Tolerance: analytical O(eps^2) bound + Monte Carlo margin.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from witness.dynamics import drift_matrix, transformation_T, simulate
from witness.partitions import apply_transformation
from battery.kuramoto import kuramoto_order_parameter, kuramoto_difference_bound


def verify_C7(
    omega_0: float = 1.0,
    k: float = 0.3,
    gamma: float = 0.1,
    beta: float = 1.0,
    t_max: float = 500.0,
    dt: float = 0.01,
    monte_carlo_margin: float = 0.05,
    seed: int = 11,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify Kuramoto near-invariance against the analytical bound."""
    rng = np.random.default_rng(seed)

    _, X = simulate(omega_0, k, gamma, beta, t_max=t_max, dt=dt, rng=rng)
    _, r_orig = kuramoto_order_parameter(X, indices=(0, 1))
    X_T = apply_transformation(X, transformation_T())
    _, r_NM = kuramoto_order_parameter(X_T, indices=(0, 1))

    diff = abs(r_orig - r_NM)
    analytical_bound = kuramoto_difference_bound(omega_0, k, gamma)
    tolerance = analytical_bound + monte_carlo_margin

    passed = diff <= tolerance

    if verbose:
        print(f"C.7 (Kuramoto near-invariance): |r_orig - r_NM| = {diff:.3e}, "
              f"analytical_bound = {analytical_bound:.3e}, "
              f"tolerance = {tolerance:.3e}, passed = {passed}")
    return {
        "passed": passed,
        "max_error": float(diff),
        "tolerance": tolerance,
        "details": {
            "r_orig": float(r_orig),
            "r_NM": float(r_NM),
            "difference": float(diff),
            "analytical_bound": analytical_bound,
            "monte_carlo_margin": monte_carlo_margin,
        },
    }


if __name__ == "__main__":
    result = verify_C7(verbose=True)
    print(f"  Passed: {result['passed']}")
