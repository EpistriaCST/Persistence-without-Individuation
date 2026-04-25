"""Transport profile: diffusion tensor and its scalar invariants.

The transport tensor D is the diffusion coefficient matrix in the Fokker-Planck
description of the dynamics. For our linear Langevin system, D = sigma * sigma^T
where sigma is the noise amplitude matrix; the Fokker-Planck operator's
diffusion term is (1/2) * sum_{ij} D_{ij} * d^2/(dx_i dx_j).

Under T in G, D transforms as D -> T*D*T^T. Its spectrum is preserved and so
are scalar invariants (trace, determinant, eigenvalues sorted, spectral
participation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def transport_eigenvalues(D: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the eigenvalues of the diffusion tensor D, sorted ascending.

    Parameters
    ----------
    D : ndarray of shape (n, n)
        Diffusion tensor.

    Returns
    -------
    eigvals : ndarray of shape (n,)
        Eigenvalues of D, sorted ascending.
    """
    eigvals = np.linalg.eigvalsh(D)  # eigvalsh for symmetric matrices
    return np.sort(eigvals)


def transport_invariants(D: NDArray[np.float64]) -> dict:
    """Return scalar invariants of the transport tensor D.

    All of these quantities are invariant under conjugation D -> T*D*T^T for
    orthogonal T (and more generally under any T that preserves the symmetry
    of D as a similarity transformation).

    Parameters
    ----------
    D : ndarray of shape (n, n)
        Diffusion tensor.

    Returns
    -------
    dict with keys:
      'trace' : sum of eigenvalues
      'det' : product of eigenvalues
      'frobenius_norm' : ||D||_F
      'eigenvalues_sorted' : eigenvalues sorted ascending
      'rank' : numerical rank (eigenvalues > 1e-12 of max)
    """
    eigvals = transport_eigenvalues(D)
    max_ev = np.max(np.abs(eigvals)) if len(eigvals) > 0 else 0.0
    rank = int(np.sum(np.abs(eigvals) > 1e-12 * max(max_ev, 1.0)))
    return {
        "trace": float(np.trace(D)),
        "det": float(np.linalg.det(D)),
        "frobenius_norm": float(np.linalg.norm(D, ord="fro")),
        "eigenvalues_sorted": eigvals,
        "rank": rank,
    }
