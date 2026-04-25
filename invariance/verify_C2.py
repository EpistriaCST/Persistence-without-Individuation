"""Verify Appendix C.2: persistence autocorrelation is exactly invariant.

For a linear observable psi(x) = v . x of state vector x, the autocorrelation
C_v(tau) = v^T * Sigma(tau) * v is exactly invariant under T in G:

    C_{Tv}(tau) = (Tv)^T * (T Sigma(tau) T^T) * (Tv)
                = v^T * Sigma(tau) * v
                = C_v(tau)

The test fixes a basis of observables {e_1, e_2, e_3, e_4} (the canonical basis
of R^4), computes C(tau) for each in the original basis, then in the
transformed basis (where the observable T*e_i is computed), and verifies that
the family of autocorrelations is preserved.

Tolerance: machine precision for the analytical comparison.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from witness.dynamics import drift_matrix, transformation_T
from witness.stationary import stationary_covariance_analytical, lagged_covariance


def verify_C2(
    omega_0: float = 1.0,
    k: float = 0.3,
    gamma: float = 0.1,
    beta: float = 1.0,
    taus: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
    tolerance: float = 1e-10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify autocorrelation invariance.

    Returns
    -------
    dict with keys 'passed', 'max_error', 'tolerance', and 'details'.
    """
    A = drift_matrix(omega_0, k, gamma)
    Sigma = stationary_covariance_analytical(omega_0, k, gamma, beta)
    T = transformation_T()

    # Sigma in the transformed basis is T * Sigma * T^T
    Sigma_T = T @ Sigma @ T.T
    A_T = T @ A @ T.T

    # Compute autocorrelations for the canonical basis observables
    n = A.shape[0]
    max_error = 0.0
    details = []
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        for tau in taus:
            # Original basis
            Sigma_tau = lagged_covariance(Sigma, A, tau)
            C_orig = float(v @ Sigma_tau @ v)
            # Transformed basis: observable is T @ v, lagged covariance is T*Sigma(tau)*T^T
            v_t = T @ v
            Sigma_tau_T = lagged_covariance(Sigma_T, A_T, tau)
            C_trans = float(v_t @ Sigma_tau_T @ v_t)
            err = abs(C_orig - C_trans)
            max_error = max(max_error, err)
            details.append({
                "observable_index": i, "tau": tau,
                "C_orig": C_orig, "C_trans": C_trans, "error": err,
            })

    passed = max_error < tolerance
    if verbose:
        print(f"C.2 (autocorrelation): max_error = {max_error:.3e}, "
              f"tolerance = {tolerance:.3e}, passed = {passed}")
    return {
        "passed": passed, "max_error": max_error,
        "tolerance": tolerance, "details": details,
    }


if __name__ == "__main__":
    result = verify_C2(verbose=True)
    print(f"  Passed: {result['passed']}")
