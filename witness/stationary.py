"""Stationary covariance and lagged covariance for the witness system.

For a linear stochastic system dx/dt = A*x + sigma*W(t) with stable A (all
eigenvalues in the open left half-plane), the stationary distribution is
Gaussian with mean zero and covariance Sigma satisfying the continuous-time
Lyapunov equation:

    A * Sigma + Sigma * A^T + D = 0

where D = sigma * sigma^T is the diffusion matrix. The lagged covariance is

    Sigma(tau) = exp(A * tau) * Sigma  for tau >= 0

The analytical formulas are well-established for linear Langevin systems; see
Gardiner (2009), Chapter 4. This module provides both the analytical solution
(via scipy.linalg.solve_continuous_lyapunov) and a numerical estimate from a
long-time Langevin simulation, for cross-verification.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from witness.dynamics import drift_matrix, diffusion_matrix, simulate


def stationary_covariance_analytical(
    omega_0: float,
    k: float,
    gamma: float,
    beta: float,
) -> NDArray[np.float64]:
    """Compute the stationary covariance Sigma analytically.

    Solves the continuous Lyapunov equation A*Sigma + Sigma*A^T + D = 0.

    For the witness system at thermal equilibrium with the bath, the analytical
    solution is the equipartition result Sigma = (1/beta) * M, where M is the
    inverse of the dimensionless mass-weighted Hessian of H_sys. This yields
    block-diagonal Sigma with the position block determined by omega_0 and k
    and the momentum block proportional to (1/beta) * I.

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

    Returns
    -------
    Sigma : ndarray of shape (4, 4)
        Stationary covariance matrix in the original basis (q1, q2, p1, p2).
    """
    A = drift_matrix(omega_0, k, gamma)
    D = diffusion_matrix(gamma, beta)
    # solve_continuous_lyapunov solves A*X + X*A^T + Q = 0 with negative-Q convention.
    # The convention requires us to pass -D so that A*Sigma + Sigma*A^T + D = 0
    # becomes A*Sigma + Sigma*A^T = -D, which is what the routine expects.
    Sigma = linalg.solve_continuous_lyapunov(A, -D)
    # Symmetrize to remove numerical asymmetry
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


def stationary_covariance_numerical(
    omega_0: float,
    k: float,
    gamma: float,
    beta: float,
    n_samples: int = 100_000,
    burn_in: int = 10_000,
    dt: float = 0.01,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Estimate the stationary covariance from a long-time Langevin simulation.

    Burns in the trajectory for `burn_in` steps to relax to the stationary
    distribution, then samples for `n_samples` steps and computes the empirical
    covariance.

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
    n_samples : int
        Number of post-burn-in steps to use for the empirical covariance.
    burn_in : int
        Number of burn-in steps.
    dt : float
        Integration timestep.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    Sigma_hat : ndarray of shape (4, 4)
        Empirical covariance estimated from the simulation.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_steps = burn_in + n_samples
    t_max = total_steps * dt
    _, X = simulate(omega_0, k, gamma, beta, t_max=t_max, dt=dt, rng=rng)
    X_post = X[burn_in:]
    Sigma_hat = np.cov(X_post.T, ddof=0)
    return Sigma_hat


def lagged_covariance(
    Sigma: NDArray[np.float64],
    A: NDArray[np.float64],
    tau: float,
) -> NDArray[np.float64]:
    """Compute the lagged covariance Sigma(tau) = exp(A*tau) * Sigma.

    Parameters
    ----------
    Sigma : ndarray of shape (n, n)
        Stationary covariance.
    A : ndarray of shape (n, n)
        Drift matrix.
    tau : float
        Time lag.

    Returns
    -------
    Sigma_tau : ndarray of shape (n, n)
        Lagged covariance.
    """
    return linalg.expm(A * tau) @ Sigma
