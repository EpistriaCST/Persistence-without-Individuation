"""Verify Appendix C.4: spectral stability is exactly invariant.

Under similarity transformation A -> T^{-1} A T (which is the action of T in G
on a linear generator), the spectrum of A is exactly preserved. Eigenvalues,
spectral gap, spectral radius, trace, and determinant are all invariant under
similarity.

Tolerance: machine precision (1e-12 for spectral comparisons).
"""

from __future__ import annotations

import numpy as np
from typing import Any

from witness.dynamics import drift_matrix, transformation_T
from battery.spectral import spectrum, spectral_invariants


def verify_C4(
    omega_0: float = 1.0,
    k: float = 0.3,
    gamma: float = 0.1,
    beta: float = 1.0,
    tolerance: float = 1e-10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify spectral-stability invariance under similarity transformation."""
    A = drift_matrix(omega_0, k, gamma)
    T = transformation_T()
    # For orthogonal T, T^{-1} = T^T, so A_T = T^T @ A @ T (similarity transform).
    # Equivalently we can use A_T = T @ A @ T.T if we view T as the change-of-basis
    # in the contravariant convention. Both yield the same spectrum.
    A_T = T.T @ A @ T

    inv_orig = spectral_invariants(A)
    inv_trans = spectral_invariants(A_T)

    # Sorted eigenvalues should agree elementwise
    eig_orig = inv_orig["eigenvalues_sorted"]
    eig_trans = inv_trans["eigenvalues_sorted"]
    eigval_error = float(np.max(np.abs(eig_orig - eig_trans)))

    scalar_errors = {
        "trace": abs(inv_orig["trace"] - inv_trans["trace"]),
        "det": abs(inv_orig["det"] - inv_trans["det"]),
        "spectral_gap": abs(inv_orig["spectral_gap"] - inv_trans["spectral_gap"]),
        "spectral_radius": abs(inv_orig["spectral_radius"] - inv_trans["spectral_radius"]),
    }
    max_error = max(eigval_error, max(scalar_errors.values()))

    passed = max_error < tolerance
    if verbose:
        print(f"C.4 (spectral stability): max_error = {max_error:.3e}, "
              f"tolerance = {tolerance:.3e}, passed = {passed}")
    return {
        "passed": passed,
        "max_error": max_error,
        "tolerance": tolerance,
        "details": {
            "eigenvalue_error": eigval_error,
            "scalar_errors": scalar_errors,
            "eigenvalues_orig": eig_orig.tolist(),
            "eigenvalues_trans": eig_trans.tolist(),
        },
    }


if __name__ == "__main__":
    result = verify_C4(verbose=True)
    print(f"  Passed: {result['passed']}")
