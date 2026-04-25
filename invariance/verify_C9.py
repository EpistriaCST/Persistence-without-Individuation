"""Verify Appendix C.9: broadcast-access (Gramian) invariants are preserved.

The controllability and observability Gramians transform under similarity:

    A -> T^{-1} A T,  B -> T^{-1} B,  C -> C T

induces

    W_c -> T^{-1} W_c T^{-T}
    W_o -> T^T W_o T

For orthogonal T, these are conjugations preserving the spectrum. All scalar
invariants of the Gramians — trace, determinant, eigenvalues, rank, Frobenius
norm — are therefore preserved exactly.

This script tests with a canonical input matrix B that distributes inputs
uniformly across the momentum components and an output matrix C that reads out
all positions, simulating a generic broadcast architecture.

Tolerance: machine precision (1e-10).
"""

from __future__ import annotations

import numpy as np
from typing import Any

from witness.dynamics import drift_matrix, transformation_T
from battery.broadcast import (
    controllability_gramian,
    observability_gramian,
    gramian_invariants,
)


def verify_C9(
    omega_0: float = 1.0,
    k: float = 0.3,
    gamma: float = 0.1,
    beta: float = 1.0,
    tolerance: float = 1e-10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify Gramian invariance under similarity."""
    A = drift_matrix(omega_0, k, gamma)
    T = transformation_T()
    A_T = T.T @ A @ T

    # Generic broadcast architecture: input enters momentum components,
    # output reads positions
    B = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    # Original basis Gramians
    Wc_orig = controllability_gramian(A, B)
    Wo_orig = observability_gramian(A, C)

    # Transformed basis: B and C transform too.
    # If state x' = T^T x, then dx'/dt = T^T A T x' + T^T B u, so B_T = T^T B.
    # Output y = C x = (C T) x', so C_T = C T.
    B_T = T.T @ B
    C_T = C @ T

    Wc_trans = controllability_gramian(A_T, B_T)
    Wo_trans = observability_gramian(A_T, C_T)

    inv_Wc_orig = gramian_invariants(Wc_orig)
    inv_Wc_trans = gramian_invariants(Wc_trans)
    inv_Wo_orig = gramian_invariants(Wo_orig)
    inv_Wo_trans = gramian_invariants(Wo_trans)

    def errors(a: dict, b: dict) -> dict[str, float]:
        return {
            "trace": abs(a["trace"] - b["trace"]),
            "det": abs(a["det"] - b["det"]),
            "frobenius_norm": abs(a["frobenius_norm"] - b["frobenius_norm"]),
            "rank_diff": abs(a["rank"] - b["rank"]),
            "eigval_max_err": float(np.max(np.abs(a["eigenvalues_sorted"]
                                                  - b["eigenvalues_sorted"]))),
        }

    err_Wc = errors(inv_Wc_orig, inv_Wc_trans)
    err_Wo = errors(inv_Wo_orig, inv_Wo_trans)
    max_error = max(max(err_Wc.values()), max(err_Wo.values()))

    passed = max_error < tolerance
    if verbose:
        print(f"C.9 (broadcast-access): max_error = {max_error:.3e}, "
              f"tolerance = {tolerance:.3e}, passed = {passed}")
    return {
        "passed": passed,
        "max_error": max_error,
        "tolerance": tolerance,
        "details": {
            "controllability_errors": err_Wc,
            "observability_errors": err_Wo,
            "Wc_eigenvalues_orig": inv_Wc_orig["eigenvalues_sorted"].tolist(),
            "Wc_eigenvalues_trans": inv_Wc_trans["eigenvalues_sorted"].tolist(),
            "Wo_eigenvalues_orig": inv_Wo_orig["eigenvalues_sorted"].tolist(),
            "Wo_eigenvalues_trans": inv_Wo_trans["eigenvalues_sorted"].tolist(),
        },
    }


if __name__ == "__main__":
    result = verify_C9(verbose=True)
    print(f"  Passed: {result['passed']}")
