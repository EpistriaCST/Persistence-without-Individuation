"""Correlation-matrix coherence: invariants of the stationary covariance Sigma.

The coherence structure of the stationary correlation matrix Sigma is captured
by a collection of scalar invariants of its spectrum:

* Trace: sum of eigenvalues
* Determinant: product of eigenvalues
* Participation ratio: PR(Sigma) = (sum lambda_i)^2 / sum lambda_i^2
* Spectral entropy: S(Sigma) = -sum p_i * log(p_i) where p_i = lambda_i / sum lambda_j

All of these are scalar functions of the eigenvalues of Sigma. Under the
similarity transformation Sigma -> T * Sigma * T^T, eigenvalues are exactly
preserved (eigenvalues of S A S^{-1} equal eigenvalues of A for any nonsingular S;
for orthogonal T, T^T = T^{-1}). Therefore all of these invariants are exactly
invariant under any T in G acting on Sigma.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def coherence_invariants(Sigma: NDArray[np.float64]) -> dict:
    """Return scalar invariants of the correlation-matrix coherence structure.

    Parameters
    ----------
    Sigma : ndarray of shape (n, n)
        Symmetric positive-(semi)definite covariance matrix.

    Returns
    -------
    dict with keys:
      'trace' : sum of eigenvalues
      'det' : product of eigenvalues
      'eigenvalues_sorted' : eigenvalues sorted ascending
      'participation_ratio' : (sum lambda_i)^2 / sum lambda_i^2
      'spectral_entropy' : -sum p_i log(p_i), p_i = lambda_i / sum lambda_j
      'frobenius_norm' : ||Sigma||_F
    """
    eigvals = np.linalg.eigvalsh(Sigma)
    eigvals = np.sort(eigvals)
    trace = float(np.sum(eigvals))
    det = float(np.prod(eigvals))
    pr = float(trace ** 2 / np.sum(eigvals ** 2)) if np.sum(eigvals ** 2) > 0 else 0.0
    # Spectral entropy with safe handling of small or zero eigenvalues
    pos = eigvals[eigvals > 1e-12]
    if len(pos) > 0:
        p = pos / np.sum(pos)
        spectral_entropy = float(-np.sum(p * np.log(p)))
    else:
        spectral_entropy = 0.0
    return {
        "trace": trace,
        "det": det,
        "eigenvalues_sorted": eigvals,
        "participation_ratio": pr,
        "spectral_entropy": spectral_entropy,
        "frobenius_norm": float(np.linalg.norm(Sigma, ord="fro")),
    }
