"""Verify Appendix C.6: phase-locking value is structurally near-invariant.

The PLV matrix is NOT in general invariant under T in G. Specifically:

* In the original basis, the PLV between (q_1, q_2) reflects the coupling-induced
  phase coherence — non-trivial off-diagonal entries.
* In the normal-mode basis, the modes are decoupled and oscillate independently;
  the off-diagonal PLV entries decay toward zero in the long-time limit.

What is preserved under T is the underlying G-invariant dynamical structure
(coupling k, damping gamma, generator spectrum). The Frobenius-norm difference
between PLV_orig and PLV_NM is bounded analytically by a function of the
dimensionless coupling alpha = k / (omega_0^2 + k); see battery.plv.
plv_difference_bound for the explicit bound.

This script computes PLV from a long Langevin trajectory in each basis and
verifies that the difference is within the analytical bound. Because PLV is
estimated from finite-length stochastic data, an empirical tolerance margin is
added to the analytical bound.

Tolerance: analytical bound + 0.05 (Monte Carlo margin from finite-trajectory PLV).
"""

from __future__ import annotations

import numpy as np
from typing import Any

from witness.dynamics import drift_matrix, transformation_T, simulate
from witness.partitions import apply_transformation
from battery.plv import phase_locking_value, plv_difference_bound


def verify_C6(
    omega_0: float = 1.0,
    k: float = 0.3,
    gamma: float = 0.1,
    beta: float = 1.0,
    t_max: float = 500.0,
    dt: float = 0.01,
    monte_carlo_margin: float = 0.05,
    seed: int = 7,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify PLV structural near-invariance against the analytical bound."""
    rng = np.random.default_rng(seed)

    _, X = simulate(omega_0, k, gamma, beta, t_max=t_max, dt=dt, rng=rng)
    # Position observables are the natural choice for PLV (the (q_1, q_2) and
    # (Q_+, Q_-) phase coherence is what frameworks track).
    PLV_orig = phase_locking_value(X, indices=(0, 1))
    X_T = apply_transformation(X, transformation_T())
    PLV_NM = phase_locking_value(X_T, indices=(0, 1))

    diff_norm = float(np.linalg.norm(PLV_orig - PLV_NM, ord="fro"))
    analytical_bound = plv_difference_bound(omega_0, k, gamma)
    tolerance = analytical_bound + monte_carlo_margin

    passed = diff_norm <= tolerance
    # PLV entries are NOT equal — they should differ. We expect the diagonal to
    # be 1 in both bases and the off-diagonals to differ.
    diff_nontrivial = diff_norm > 1e-6

    if verbose:
        print(f"C.6 (PLV near-invariance): ||PLV_orig - PLV_NM||_F = {diff_norm:.3e}, "
              f"analytical_bound = {analytical_bound:.3e}, "
              f"tolerance = {tolerance:.3e}, passed = {passed}")
    return {
        "passed": passed,
        "max_error": diff_norm,
        "tolerance": tolerance,
        "details": {
            "diff_frobenius_norm": diff_norm,
            "analytical_bound": analytical_bound,
            "monte_carlo_margin": monte_carlo_margin,
            "PLV_orig": PLV_orig.tolist(),
            "PLV_NM": PLV_NM.tolist(),
            "structural_difference_confirmed": diff_nontrivial,
        },
    }


if __name__ == "__main__":
    result = verify_C6(verbose=True)
    print(f"  Passed: {result['passed']}")
