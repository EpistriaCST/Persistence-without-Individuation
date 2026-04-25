"""Test suite for the persistence-without-individuation reference implementation.

The test suite verifies the numerical claims of §E.5 of the paper:

  1. Stationary covariance: analytical solution agrees with empirical covariance
     from a long-time Langevin simulation to within 1% relative error.
  2. Spectrum of the generator agrees between the original basis and the
     normal-mode basis to machine precision.
  3. Correlation-matrix coherence invariants — trace, eigenvalues, participation
     ratio, spectral entropy — agree exactly between bases.
  4. PLV matrix entries differ between bases (PLV_orig != PLV_NM), and the
     Frobenius-norm difference is within the analytically-predicted structural
     bound.
  5. Kuramoto r agrees between bases to within O((k/omega_0^2)^2).
  6. The witness's identity-set Ind(P_1) differs from Ind(P_2) at the set level
     under each framework's systemhood criterion.

Run with:
    pytest

or for verbose output:
    pytest -v
"""
