"""Return and overlap statistics.

For a measurable region R in state space, the return probability after time T is

    P_return(R, T) = P(x(T) in R | x(0) in R, x distributed as rho_inf)

For a linear-Gaussian system at stationarity, both x(0) and x(T) are jointly
Gaussian with covariance Sigma and lagged covariance Sigma(T). The return
probability for a region R is therefore a Gaussian-orthant integral.

Under T in G, regions transform as R -> T(R), the lagged covariance transforms
as Sigma(T) -> T*Sigma(T)*T^T, and the joint distribution of (x(0), x(T))
transforms covariantly. The family of return statistics across all regions is
therefore preserved: the return probability for R in basis B equals the return
probability for T(R) in basis T*B.

This module computes return probabilities for ellipsoidal regions defined by
quadratic forms x^T M x <= r^2, since these are tractable closed-form for
Gaussian distributions and form a natural family invariant under linear
transformations of the state space.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats


def return_probability_basis(
    A: NDArray[np.float64],
    Sigma: NDArray[np.float64],
    M: NDArray[np.float64],
    r: float,
    tau: float,
    n_samples: int = 50_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate P(x(tau) in R | x(0) in R) for the ellipsoidal region
    R = {x : x^T M x <= r^2}.

    Uses Monte Carlo sampling from the Gaussian stationary distribution and the
    conditional Gaussian for x(tau) | x(0).

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Drift matrix.
    Sigma : ndarray of shape (n, n)
        Stationary covariance.
    M : ndarray of shape (n, n)
        Symmetric positive-definite matrix defining the region.
    r : float
        Region radius.
    tau : float
        Time lag.
    n_samples : int
        Monte Carlo sample size.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    P : float
        Estimated return probability.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = Sigma.shape[0]
    # Sample x(0) from the stationary distribution
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(n))
    z = rng.standard_normal((n_samples, n))
    X0 = z @ L.T

    # Condition on x(0) in R
    quadratic_form_0 = np.einsum("ij,jk,ik->i", X0, M, X0)
    in_R_0 = quadratic_form_0 <= r ** 2
    if not np.any(in_R_0):
        return 0.0

    X0_in = X0[in_R_0]
    # Conditional mean of x(tau) given x(0): exp(A*tau) * x(0)
    # Conditional covariance: Sigma - exp(A*tau) * Sigma * exp(A*tau)^T
    expAt = linalg.expm(A * tau)
    cond_means = X0_in @ expAt.T
    cond_cov = Sigma - expAt @ Sigma @ expAt.T
    # Symmetrize
    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    # Add small regularization for numerical stability
    cond_cov = cond_cov + 1e-10 * np.eye(n)
    L_cond = np.linalg.cholesky(cond_cov)

    # Sample x(tau)
    z_t = rng.standard_normal((len(X0_in), n))
    Xt = cond_means + z_t @ L_cond.T

    # Check region membership at tau
    quadratic_form_t = np.einsum("ij,jk,ik->i", Xt, M, Xt)
    in_R_t = quadratic_form_t <= r ** 2

    return float(np.mean(in_R_t))


def transformed_region(
    M: NDArray[np.float64],
    T: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the matrix M' defining the transformed region T(R).

    If R = {x : x^T M x <= r^2} then T(R) = {y : y^T M' y <= r^2} with
    M' = T * M * T^T.

    Parameters
    ----------
    M : ndarray of shape (n, n)
        Region-defining matrix.
    T : ndarray of shape (n, n)
        Linear transformation.

    Returns
    -------
    M_prime : ndarray of shape (n, n)
        Region-defining matrix in the transformed basis.
    """
    # If x = T*x_orig, then x^T M x = (T*x_orig)^T M (T*x_orig) = x_orig^T (T^T M T) x_orig.
    # The region in the original basis is x_orig^T M' x_orig <= r^2 with M' = T^T M T.
    # Equivalently, expressed in the transformed basis where y = T*x_orig, the region is
    # y^T (T M T^T) y <= r^2 because (T^{-T} y)^T M (T^{-T} y) = y^T (T^{-1})^T M T^{-T} y;
    # for orthogonal T, T^{-1} = T^T, so this gives T M T^T.
    return T @ M @ T.T
