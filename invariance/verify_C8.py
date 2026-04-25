"""Verify Appendix C.8: correlation-matrix coherence invariants are exactly preserved.

Under Sigma -> T * Sigma * T^T for orthogonal T, the eigenvalues of Sigma are
exactly preserved. All scalar functions of the spectrum — trace, determinant,
participation ratio, spectral entropy, Frobenius norm — are therefore exactly
invariant.

Tolerance: machine precision (1e-10).
"""

from __future__ import annotations

import numpy as np
from typing import Any

from witness.dynamics import transformation_T
from witness.stationary import stationary_covariance_analytical
from battery.coherence import coherence_invariants


def verify_C8(
    omega_0: float = 1.0,
    k: float = 0.3,
    gamma: float = 0.1,
    beta: float = 1.0,
    tolerance: float = 1e-10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify correlation-matrix coherence invariance under Sigma -> T Sigma T^T."""
    Sigma = stationary_covariance_analytical(omega_0, k, gamma, beta)
    T = transformation_T()
    Sigma_T = T @ Sigma @ T.T

    inv_orig = coherence_invariants(Sigma)
    inv_trans = coherence_invariants(Sigma_T)

    eigval_error = float(np.max(np.abs(inv_orig["eigenvalues_sorted"]
                                       - inv_trans["eigenvalues_sorted"])))
    scalar_errors = {
        "trace": abs(inv_orig["trace"] - inv_trans["trace"]),
        "det": abs(inv_orig["det"] - inv_trans["det"]),
        "participation_ratio": abs(inv_orig["participation_ratio"]
                                   - inv_trans["participation_ratio"]),
        "spectral_entropy": abs(inv_orig["spectral_entropy"]
                                - inv_trans["spectral_entropy"]),
        "frobenius_norm": abs(inv_orig["frobenius_norm"]
                              - inv_trans["frobenius_norm"]),
    }
    max_error = max(eigval_error, max(scalar_errors.values()))

    passed = max_error < tolerance
    if verbose:
        print(f"C.8 (correlation-matrix coherence): max_error = {max_error:.3e}, "
              f"tolerance = {tolerance:.3e}, passed = {passed}")
    return {
        "passed": passed,
        "max_error": max_error,
        "tolerance": tolerance,
        "details": {
            "eigenvalue_error": eigval_error,
            "scalar_errors": scalar_errors,
            "eigenvalues_orig": inv_orig["eigenvalues_sorted"].tolist(),
            "eigenvalues_trans": inv_trans["eigenvalues_sorted"].tolist(),
        },
    }


if __name__ == "__main__":
    result = verify_C8(verbose=True)
    print(f"  Passed: {result['passed']}")
