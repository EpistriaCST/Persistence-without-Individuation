"""Phase-locking value: pairwise phase coherence among oscillatory observables.

The PLV between observables psi_i and psi_j is

    PLV_{ij} = | <exp(i * (phi_i(t) - phi_j(t)))> |

where phi_i is the instantaneous phase of psi_i (extracted via the Hilbert
transform of the trajectory).

Under T in G, the entries of the PLV matrix are NOT in general preserved: the
PLV is a phase coherence measure of specific observables, and if the observables
themselves transform under T, their phases change. What is preserved is the
underlying G-invariant dynamical structure of the substrate (coupling k,
damping gamma, noise spectrum, generator spectrum); PLV depends on these
parameters but the entry-by-entry matrix is not invariant.

This module computes PLV from a trajectory and provides a near-invariance
diagnostic: the Frobenius norm of the difference between PLV in basis 1 and
basis 2 is bounded by an analytical function of the system parameters in the
weak-coupling regime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert


def phase_locking_value(
    X: NDArray[np.float64],
    indices: list[int] | tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    """Compute the pairwise PLV matrix from a trajectory.

    Parameters
    ----------
    X : ndarray of shape (N, n)
        Trajectory.
    indices : sequence of ints, optional
        Indices of observables to use. Defaults to all components.

    Returns
    -------
    PLV : ndarray of shape (m, m)
        Pairwise PLV matrix where m = len(indices).
    """
    if indices is None:
        indices = tuple(range(X.shape[1]))
    indices = list(indices)
    m = len(indices)

    # Hilbert transform along the time axis to extract instantaneous phases
    analytic = hilbert(X[:, indices], axis=0)
    phases = np.angle(analytic)

    PLV = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            phase_diff = phases[:, i] - phases[:, j]
            PLV[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
    return PLV


def plv_difference_bound(
    omega_0: float,
    k: float,
    gamma: float,
) -> float:
    """Return an analytical upper bound on the Frobenius norm of PLV_orig - PLV_NM.

    For the linear-Gaussian witness in the weak-coupling regime, the difference
    in PLV matrices between the original and normal-mode bases is bounded by a
    function of the dimensionless coupling alpha = k / (omega_0^2 + k). Specifically,
    the off-diagonal PLV entries in the original basis are at most alpha, while
    they are zero in the normal-mode basis (the modes are decoupled). The bound
    here is conservative.

    Parameters
    ----------
    omega_0 : float
        Natural frequency.
    k : float
        Coupling strength.
    gamma : float
        Damping coefficient.

    Returns
    -------
    bound : float
        Upper bound on ||PLV_orig - PLV_NM||_F.
    """
    alpha = k / (omega_0 ** 2 + k)
    # Frobenius norm: 4 off-diagonal entries each bounded by alpha;
    # diagonal entries are 1 in both bases. Bound: sqrt(4) * alpha = 2 * alpha.
    return float(2.0 * alpha)
