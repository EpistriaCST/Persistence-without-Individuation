"""Verify Appendix C.3: return statistics are preserved under region transformation.

The return probability P(R, tau) = P(x(tau) in R | x(0) in R) is determined by
the transition density p, which is a function of L alone. Under T in G, regions
transform as R -> T(R), and the family of return statistics across all regions
is preserved:

    P_orig(R, tau) = P_trans(T(R), tau)

The test computes P(R, tau) for a small family of ellipsoidal regions in the
original basis and the corresponding transformed regions in the normal-mode
basis, then verifies agreement to within Monte Carlo sampling error.

Tolerance: 3 standard errors of the Monte Carlo estimate (each estimate uses
n_samples = 50_000 by default).
"""

from __future__ import annotations

import numpy as np
from typing import Any

from witness.dynamics import drift_matrix, transformation_T
from witness.stationary import stationary_covariance_analytical
from battery.returns import return_probability_basis, transformed_region


def verify_C3(
    omega_0: float = 1.0,
    k: float = 0.3,
    gamma: float = 0.1,
    beta: float = 1.0,
    taus: tuple[float, ...] = (0.5, 1.0, 2.0),
    n_samples: int = 50_000,
    tolerance: float = 0.02,  # ~3 sigma at n_samples = 50k
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify return-statistics invariance up to region transformation."""
    A = drift_matrix(omega_0, k, gamma)
    Sigma = stationary_covariance_analytical(omega_0, k, gamma, beta)
    T = transformation_T()
    A_T = T @ A @ T.T
    Sigma_T = T @ Sigma @ T.T

    rng_orig = np.random.default_rng(seed)
    rng_trans = np.random.default_rng(seed + 1)

    n = A.shape[0]
    # Three test regions: the unit ball, an aligned ellipsoid, and a tilted ellipsoid
    M_unit = np.eye(n)
    M_aligned = np.diag([1.0, 2.0, 0.5, 1.5])
    # A SPD matrix for the tilted ellipsoid
    rng_M = np.random.default_rng(123)
    G_random = rng_M.standard_normal((n, n))
    M_tilted = G_random.T @ G_random + 0.1 * np.eye(n)
    M_tilted = 0.5 * (M_tilted + M_tilted.T)

    test_regions = [("unit", M_unit, 1.5),
                    ("aligned", M_aligned, 1.5),
                    ("tilted", M_tilted, 2.0)]

    max_error = 0.0
    details = []
    for region_name, M, r in test_regions:
        for tau in taus:
            P_orig = return_probability_basis(A, Sigma, M, r, tau,
                                              n_samples=n_samples, rng=rng_orig)
            M_trans = transformed_region(M, T)
            P_trans = return_probability_basis(A_T, Sigma_T, M_trans, r, tau,
                                               n_samples=n_samples, rng=rng_trans)
            err = abs(P_orig - P_trans)
            max_error = max(max_error, err)
            details.append({
                "region": region_name, "tau": tau,
                "P_orig": P_orig, "P_trans": P_trans, "error": err,
            })

    passed = max_error < tolerance
    if verbose:
        print(f"C.3 (return statistics): max_error = {max_error:.3e}, "
              f"tolerance = {tolerance:.3e}, passed = {passed}")
    return {
        "passed": passed, "max_error": max_error,
        "tolerance": tolerance, "details": details,
    }


if __name__ == "__main__":
    result = verify_C3(verbose=True)
    print(f"  Passed: {result['passed']}")
