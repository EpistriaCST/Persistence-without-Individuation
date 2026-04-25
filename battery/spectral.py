"""Spectral stability: eigenvalues, eigenvectors, and stability margins of L.

The spectral structure of the dynamics is characterized by the spectrum of the
generator L (the drift matrix A in the linear case). Under similarity
transformation L -> T^{-1} L T (which is the action of T in G on a linear
generator), the spectrum is preserved. Eigenvalues, eigenvectors (up to basis
relabeling), spectral gap, and any scalar invariant of the spectrum are all
exactly invariant under similarity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def spectrum(A: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return the eigenvalues of A, sorted by real part then imaginary part."""
    eigvals = np.linalg.eigvals(A)
    # Sort for deterministic output (real part ascending, then imag part ascending)
    order = np.lexsort((eigvals.imag, eigvals.real))
    return eigvals[order]


def spectral_gap(A: NDArray[np.float64]) -> float:
    """Return the spectral gap: min(|Re(lambda_i)|) over eigenvalues lambda_i.

    For a stable linear system (all eigenvalues with negative real part), the
    spectral gap quantifies the slowest decay rate.
    """
    eigvals = np.linalg.eigvals(A)
    return float(np.min(np.abs(eigvals.real)))


def spectral_invariants(A: NDArray[np.float64]) -> dict:
    """Return a dictionary of scalar invariants of the spectrum of A.

    All of these quantities are invariant under similarity transformations
    L -> T^{-1} L T and therefore under any T in G acting on a linear generator.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Drift matrix or generator.

    Returns
    -------
    dict with keys:
      'trace' : sum of eigenvalues
      'det' : product of eigenvalues
      'spectral_gap' : minimum absolute value of the real parts
      'spectral_radius' : maximum absolute value of the eigenvalues
      'eigenvalues_sorted' : sorted eigenvalues (complex)
    """
    eigvals = spectrum(A)
    return {
        "trace": float(np.trace(A).real),
        "det": float(np.linalg.det(A).real),
        "spectral_gap": float(np.min(np.abs(eigvals.real))),
        "spectral_radius": float(np.max(np.abs(eigvals))),
        "eigenvalues_sorted": eigvals,
    }
