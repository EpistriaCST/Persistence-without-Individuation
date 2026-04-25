"""Broadcast-access and functional availability via controllability and
observability Gramians.

Global Workspace Theory characterizes a system through its broadcast structure:
which degrees of freedom can propagate information to which others, and which
admit external readout. In control-theoretic terms, this is captured by the
controllability and observability Gramians.

For a linear system dx/dt = A*x + B*u with output y = C*x and a stable A, the
infinite-horizon controllability Gramian W_c and observability Gramian W_o
satisfy:

    A * W_c + W_c * A^T + B * B^T = 0
    A^T * W_o + W_o * A + C^T * C = 0

Their rank, spectrum, and scalar invariants encode the system's broadcast
structure. Under the similarity transformation A -> T^{-1} A T, B -> T^{-1} B,
C -> C T (which is the change-of-basis induced by T in G), the Gramians
transform as W_c -> T^{-1} W_c T^{-T} and W_o -> T^T W_o T. Their spectra are
preserved, as are all scalar invariants thereof.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def controllability_gramian(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the infinite-horizon controllability Gramian.

    Solves A * W_c + W_c * A^T + B * B^T = 0 via the continuous-time Lyapunov
    equation.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Drift matrix (must be stable: all eigenvalues with negative real part).
    B : ndarray of shape (n, m)
        Input matrix.

    Returns
    -------
    W_c : ndarray of shape (n, n)
        Controllability Gramian.
    """
    Q = -B @ B.T
    W = linalg.solve_continuous_lyapunov(A, Q)
    # Symmetrize
    W = 0.5 * (W + W.T)
    return W


def observability_gramian(
    A: NDArray[np.float64],
    C: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the infinite-horizon observability Gramian.

    Solves A^T * W_o + W_o * A + C^T * C = 0.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Drift matrix (must be stable).
    C : ndarray of shape (p, n)
        Output matrix.

    Returns
    -------
    W_o : ndarray of shape (n, n)
        Observability Gramian.
    """
    Q = -C.T @ C
    W = linalg.solve_continuous_lyapunov(A.T, Q)
    W = 0.5 * (W + W.T)
    return W


def gramian_invariants(W: NDArray[np.float64]) -> dict:
    """Return scalar invariants of a Gramian.

    Parameters
    ----------
    W : ndarray of shape (n, n)
        Symmetric positive-semidefinite Gramian.

    Returns
    -------
    dict with keys:
      'trace' : sum of eigenvalues
      'det' : product of eigenvalues
      'eigenvalues_sorted' : eigenvalues sorted ascending
      'rank' : numerical rank
      'frobenius_norm' : ||W||_F
    """
    eigvals = np.linalg.eigvalsh(W)
    eigvals = np.sort(eigvals)
    max_ev = np.max(np.abs(eigvals)) if len(eigvals) > 0 else 0.0
    rank = int(np.sum(np.abs(eigvals) > 1e-10 * max(max_ev, 1.0)))
    return {
        "trace": float(np.sum(eigvals)),
        "det": float(np.prod(eigvals)),
        "eigenvalues_sorted": eigvals,
        "rank": rank,
        "frobenius_norm": float(np.linalg.norm(W, ord="fro")),
    }
