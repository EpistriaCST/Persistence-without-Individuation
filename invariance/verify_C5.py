"""Verify Appendix C.5: transport tensor invariants are exactly preserved.

Under T in G, the diffusion tensor transforms as D -> T*D*T^T. For orthogonal T
(the case relevant to the witness), the spectrum of D and all of its scalar
invariants — trace, determinant, eigenvalues, Frobenius norm — are preserved
exactly.

Tolerance: machine precision (1e-10).
"""

from __future__ import annotations

import numpy as np
from typing import Any

from witness.dynamics import diffusion_matrix, transformation_T
from battery.transport import transport_invariants


def verify_C5(
    omega_0: float = 1.0,
    k: float = 0.3,
    gamma: float = 0.1,
    beta: float = 1.0,
    tolerance: float = 1e-10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify transport-profile invariance under T D T^T."""
    D = diffusion_matrix(gamma, beta)
    T = transformation_T()
    D_T = T @ D @ T.T

    inv_orig = transport_invariants(D)
    inv_trans = transport_invariants(D_T)

    eigval_error = float(np.max(np.abs(inv_orig["eigenvalues_sorted"]
                                       - inv_trans["eigenvalues_sorted"])))
    scalar_errors = {
        "trace": abs(inv_orig["trace"] - inv_trans["trace"]),
        "det": abs(inv_orig["det"] - inv_trans["det"]),
        "frobenius_norm": abs(inv_orig["frobenius_norm"] - inv_trans["frobenius_norm"]),
        "rank_diff": abs(inv_orig["rank"] - inv_trans["rank"]),
    }
    max_error = max(eigval_error, max(scalar_errors.values()))

    passed = max_error < tolerance
    if verbose:
        print(f"C.5 (transport profile): max_error = {max_error:.3e}, "
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
    result = verify_C5(verbose=True)
    print(f"  Passed: {result['passed']}")
