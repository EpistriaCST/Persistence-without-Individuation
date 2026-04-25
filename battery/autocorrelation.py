"""Persistence autocorrelation: C_v(tau) = <psi(x(t)) psi(x(t+tau))>.

For a linear observable psi(x) = v . x of state vector x, the autocorrelation at
stationarity is

    C_v(tau) = v^T * Sigma(tau) * v

where Sigma(tau) = exp(A*tau) * Sigma is the lagged covariance. C_v(tau) is
determined entirely by L (the generator) and Sigma; under T in G the matrix
Sigma(tau) transforms as T*Sigma(tau)*T^T while v transforms as T*v, so
v^T * Sigma(tau) * v is invariant.

This module provides both an analytical computation (from L and Sigma) and a
trajectory-based estimator (for cross-verification against the analytical form).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def autocorrelation(
    A: NDArray[np.float64],
    Sigma: NDArray[np.float64],
    v: NDArray[np.float64],
    tau: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Compute C_v(tau) = v^T * Sigma(tau) * v analytically.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Drift matrix.
    Sigma : ndarray of shape (n, n)
        Stationary covariance.
    v : ndarray of shape (n,)
        Coefficient vector defining the observable psi(x) = v . x.
    tau : float or ndarray of shape (m,)
        Time lag(s) at which to evaluate the autocorrelation.

    Returns
    -------
    C : float or ndarray of shape (m,)
        Autocorrelation values at the requested lag(s).
    """
    if np.isscalar(tau):
        Sigma_tau = linalg.expm(A * float(tau)) @ Sigma
        return float(v @ Sigma_tau @ v)
    taus = np.asarray(tau, dtype=np.float64)
    out = np.empty_like(taus)
    for i, t in enumerate(taus):
        Sigma_tau = linalg.expm(A * float(t)) @ Sigma
        out[i] = v @ Sigma_tau @ v
    return out


def autocorrelation_from_trajectory(
    X: NDArray[np.float64],
    v: NDArray[np.float64],
    lags: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Estimate C_v at a sequence of integer lags from a trajectory.

    Computes psi(t) = v . X[t] and the empirical autocorrelation
    (1/(N-l)) * sum_t psi(t) * psi(t+l) for each lag l in `lags`.

    Parameters
    ----------
    X : ndarray of shape (N, n)
        Trajectory; each row is a state vector.
    v : ndarray of shape (n,)
        Coefficient vector.
    lags : ndarray of shape (m,) of ints
        Lag indices.

    Returns
    -------
    C_hat : ndarray of shape (m,)
        Empirical autocorrelations.
    """
    psi = X @ v
    N = len(psi)
    C_hat = np.empty(len(lags), dtype=np.float64)
    for i, l in enumerate(lags):
        l = int(l)
        if l == 0:
            C_hat[i] = float(np.mean(psi * psi))
        else:
            C_hat[i] = float(np.mean(psi[:-l] * psi[l:]))
    return C_hat
