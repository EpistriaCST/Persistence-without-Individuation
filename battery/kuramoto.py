"""Kuramoto order parameter: ensemble phase coherence.

The Kuramoto order parameter is

    r * exp(i * psi) = (1/N) * sum_k exp(i * phi_k)

where phi_k is the instantaneous phase of oscillator k and r in [0, 1] measures
global phase coherence. r = 1 corresponds to all oscillators in phase; r = 0
corresponds to incoherent phases.

For the two-oscillator witness, the Kuramoto r is

    r = | (exp(i*phi_1) + exp(i*phi_2)) / 2 | = | cos((phi_1 - phi_2) / 2) |

In the original basis with coupling k, the time-averaged r reflects the typical
phase difference between the two oscillators, which is set by the coupling
strength. In the normal-mode basis the modes are decoupled and oscillate at
different frequencies, so the time-averaged r tends to a different value.

The Kuramoto order parameter, like PLV, is structurally near-invariant: the
underlying dynamical parameters that determine it are preserved under T in G,
but the entry-wise value in each basis depends on which observables are paired.
The near-invariance is at order O((k/omega_0^2)^2) in the weak-coupling limit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert


def kuramoto_order_parameter(
    X: NDArray[np.float64],
    indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Compute the time-resolved and time-averaged Kuramoto order parameter.

    Parameters
    ----------
    X : ndarray of shape (N, n)
        Trajectory.
    indices : sequence of ints, optional
        Indices of observables to use. Defaults to all components.

    Returns
    -------
    r_t : ndarray of shape (N,)
        Time-resolved Kuramoto order parameter.
    r_mean : float
        Time-averaged Kuramoto order parameter.
    """
    if indices is None:
        indices = tuple(range(X.shape[1]))
    indices = list(indices)

    analytic = hilbert(X[:, indices], axis=0)
    phases = np.angle(analytic)
    z = np.mean(np.exp(1j * phases), axis=1)
    r_t = np.abs(z)
    r_mean = float(np.mean(r_t))
    return r_t, r_mean


def kuramoto_difference_bound(
    omega_0: float,
    k: float,
    gamma: float,
) -> float:
    """Return an analytical upper bound on |r_orig - r_NM|.

    In the weak-coupling regime, the difference between the time-averaged
    Kuramoto order parameter in the original basis and the normal-mode basis
    is O((k / omega_0^2)^2). The bound here is the leading-order coefficient.

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
        Upper bound on |r_orig - r_NM|.
    """
    eps = k / (omega_0 ** 2)
    # Leading-order bound is O(eps^2); we use a generous prefactor of 2.
    return float(2.0 * eps ** 2)
