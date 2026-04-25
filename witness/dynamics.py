"""Langevin SDE dynamics for the two-oscillator witness substrate.

The witness substrate consists of two coupled harmonic oscillators of unit mass with
positions q1, q2 and conjugate momenta p1, p2, natural frequency omega_0, coupling
constant k, damping gamma, and inverse temperature beta:

    H_sys = (1/2)(p1^2 + p2^2) + (1/2)*omega_0^2*(q1^2 + q2^2) + (1/2)*k*(q1 - q2)^2

The system is coupled to independent Langevin baths through dissipative terms:

    dq_i/dt = p_i
    dp_i/dt = -dH_sys/dq_i - gamma*p_i + xi_i(t)

where xi_i are independent Gaussian white-noise processes with
<xi_i(t) xi_j(t')> = 2*gamma*beta^(-1) * delta_ij * delta(t - t').

In state-vector form x = (q1, q2, p1, p2), the dynamics are linear:

    dx/dt = A*x + sigma*W(t)

with drift matrix A and diffusion matrix D = sigma*sigma^T as specified below.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional


def drift_matrix(omega_0: float, k: float, gamma: float) -> NDArray[np.float64]:
    """Return the 4x4 drift matrix A for the witness system.

    State vector ordering: x = (q1, q2, p1, p2).

    Parameters
    ----------
    omega_0 : float
        Natural frequency of the uncoupled oscillators.
    k : float
        Coupling strength.
    gamma : float
        Damping coefficient.

    Returns
    -------
    A : ndarray of shape (4, 4)
        Drift matrix such that the deterministic part of the dynamics is dx/dt = A*x.
    """
    w2 = omega_0 ** 2
    return np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(w2 + k), k, -gamma, 0.0],
            [k, -(w2 + k), 0.0, -gamma],
        ],
        dtype=np.float64,
    )


def diffusion_matrix(gamma: float, beta: float) -> NDArray[np.float64]:
    """Return the 4x4 diffusion matrix D for the witness system.

    The noise enters only through the momentum components, so D is block-diagonal
    with a zero block on the position degrees of freedom and a 2*gamma/beta block
    on the momentum degrees of freedom.

    Parameters
    ----------
    gamma : float
        Damping coefficient.
    beta : float
        Inverse temperature (k_B*T)^(-1).

    Returns
    -------
    D : ndarray of shape (4, 4)
        Diffusion matrix such that dx = A*x*dt + sigma*dW with D = sigma*sigma^T.
    """
    D = np.zeros((4, 4), dtype=np.float64)
    D[2, 2] = 2.0 * gamma / beta
    D[3, 3] = 2.0 * gamma / beta
    return D


def transformation_T() -> NDArray[np.float64]:
    """Return the orthogonal transformation T relating P1 to P2.

    T maps the original-basis state vector x = (q1, q2, p1, p2) to the
    normal-mode basis state vector x' = (Q+, Q-, P+, P-) where

        Q+ = (q1 + q2)/sqrt(2),  Q- = (q1 - q2)/sqrt(2),
        P+ = (p1 + p2)/sqrt(2),  P- = (p1 - p2)/sqrt(2).

    T is orthogonal: T*T^T = I, det(T) = 1.

    Returns
    -------
    T : ndarray of shape (4, 4)
        Orthogonal block-diagonal transformation acting on (q1, q2) and (p1, p2)
        separately by the same 2x2 rotation.
    """
    s = 1.0 / np.sqrt(2.0)
    block = np.array([[s, s], [s, -s]], dtype=np.float64)
    T = np.zeros((4, 4), dtype=np.float64)
    T[0:2, 0:2] = block
    T[2:4, 2:4] = block
    return T


def simulate(
    omega_0: float,
    k: float,
    gamma: float,
    beta: float,
    t_max: float,
    dt: float,
    x0: Optional[NDArray[np.float64]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate the Langevin SDE using exact discrete-time OU sampling.

    For a linear Langevin system dx = A*x*dt + sigma*dW, the exact discrete
    update over an interval dt is

        x(t + dt) = F * x(t) + xi

    where F = exp(A*dt) and xi is a zero-mean Gaussian with covariance
    Q(dt) = Sigma_inf - F * Sigma_inf * F^T, and Sigma_inf is the stationary
    covariance. This sampler is exact in the weak sense: the discrete-time
    moments of the trajectory match the continuous-time moments exactly, with
    no O(dt) bias of the kind incurred by the Euler-Maruyama scheme.

    See Gardiner (2009), Chapter 4, for the derivation. The advantage over
    Euler-Maruyama is precisely that the empirical moments converge to the
    analytical stationary distribution as the trajectory length grows, rather
    than to a slightly biased fixed point.

    Parameters
    ----------
    omega_0 : float
        Natural frequency.
    k : float
        Coupling strength.
    gamma : float
        Damping coefficient.
    beta : float
        Inverse temperature.
    t_max : float
        Total simulation time.
    dt : float
        Sampling interval (need not be small relative to 1/omega_0 since the
        integration is exact). Smaller dt yields finer time-resolution but does
        not affect bias.
    x0 : ndarray of shape (4,), optional
        Initial state. Defaults to the origin.
    rng : numpy.random.Generator, optional
        Random number generator. Defaults to numpy.random.default_rng().

    Returns
    -------
    t : ndarray of shape (n_steps + 1,)
        Time grid.
    X : ndarray of shape (n_steps + 1, 4)
        Trajectory; X[i] is the state at time t[i].
    """
    from scipy import linalg
    # Imported here to avoid a top-level dependency on scipy where not needed.
    from witness.stationary import stationary_covariance_analytical

    if rng is None:
        rng = np.random.default_rng()
    if x0 is None:
        x0 = np.zeros(4, dtype=np.float64)

    A = drift_matrix(omega_0, k, gamma)

    # Compute the exact one-step propagator F = exp(A*dt) and the conditional
    # noise covariance Q(dt) = Sigma_inf - F * Sigma_inf * F^T.
    F = linalg.expm(A * dt)
    Sigma_inf = stationary_covariance_analytical(omega_0, k, gamma, beta)
    Q = Sigma_inf - F @ Sigma_inf @ F.T
    Q = 0.5 * (Q + Q.T)
    # Add minimal regularization for the Cholesky decomposition; numerical
    # asymmetry can otherwise produce a non-PD Q that fails Cholesky.
    Q = Q + 1e-14 * np.eye(4)
    L = np.linalg.cholesky(Q)

    n_steps = int(round(t_max / dt))
    t = np.arange(n_steps + 1, dtype=np.float64) * dt
    X = np.empty((n_steps + 1, 4), dtype=np.float64)
    X[0] = x0

    for i in range(n_steps):
        z = rng.standard_normal(4)
        X[i + 1] = F @ X[i] + L @ z

    return t, X
