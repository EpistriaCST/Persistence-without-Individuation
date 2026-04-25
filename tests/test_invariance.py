"""Test suite for the eight invariance verifications of §4.2 and Appendix C.

These tests exercise the verify_CN scripts in the invariance package and
assert that each invariance claim holds at its specified tolerance. Failures
of these tests would indicate a regression in the implementation that
contradicts the paper's analytical results.
"""

import numpy as np
import pytest

from invariance import (
    verify_C2, verify_C3, verify_C4, verify_C5,
    verify_C6, verify_C7, verify_C8, verify_C9,
)


# ----- Exact-invariance claims (machine precision) -----

def test_C2_autocorrelation_invariant(witness_params):
    """§C.2: persistence autocorrelation is exactly G-invariant."""
    result = verify_C2(**witness_params)
    assert result["passed"], (
        f"C.2 failed: max_error = {result['max_error']:.3e}, "
        f"tolerance = {result['tolerance']:.3e}"
    )


def test_C4_spectral_stability_invariant(witness_params):
    """§C.4: spectrum of the generator is exactly preserved under similarity."""
    result = verify_C4(**witness_params)
    assert result["passed"], f"C.4 failed: {result}"


def test_C5_transport_invariant(witness_params):
    """§C.5: transport tensor invariants are exactly preserved."""
    result = verify_C5(**witness_params)
    assert result["passed"], f"C.5 failed: {result}"


def test_C8_coherence_invariants_exact(witness_params):
    """§C.8: correlation-matrix coherence invariants are exactly preserved."""
    result = verify_C8(**witness_params)
    assert result["passed"], f"C.8 failed: {result}"


def test_C9_broadcast_invariant(witness_params):
    """§C.9: Gramian invariants are preserved under similarity."""
    result = verify_C9(**witness_params)
    assert result["passed"], f"C.9 failed: {result}"


# ----- Monte-Carlo / statistical claims -----

def test_C3_returns_invariant_up_to_region_transformation(witness_params):
    """§C.3: return statistics for region R agree with statistics for T(R).

    Tolerance allows for Monte Carlo sampling error at n_samples = 50_000.
    """
    result = verify_C3(**witness_params)
    assert result["passed"], f"C.3 failed: {result}"


# ----- Near-invariance claims (PLV and Kuramoto) -----

def test_C6_plv_within_analytical_bound(witness_params):
    """§C.6: PLV difference between bases is within the analytical structural bound.

    The PLV matrix differs entry-wise between the original and normal-mode bases
    (this is the structural near-invariance result of §4.2). The Frobenius-norm
    difference is bounded analytically by 2 * alpha where alpha = k / (omega_0^2 + k).
    """
    result = verify_C6(**witness_params)
    assert result["passed"], (
        f"C.6 failed: ||PLV_orig - PLV_NM||_F = {result['max_error']:.3e}, "
        f"analytical bound + MC margin = {result['tolerance']:.3e}"
    )
    # And PLV is actually different — confirms structural rather than exact invariance
    assert result["details"]["structural_difference_confirmed"], (
        "PLV matrices were unexpectedly equal; near-invariance is the expected regime"
    )


def test_C7_kuramoto_within_analytical_bound(witness_params):
    """§C.7: Kuramoto r difference is within O((k/omega_0^2)^2) bound.

    The time-averaged Kuramoto order parameter differs between bases at order
    O((k/omega_0^2)^2). At k = 0.3 and omega_0 = 1, the leading-order bound is
    2 * (0.3)^2 = 0.18.
    """
    result = verify_C7(**witness_params)
    assert result["passed"], (
        f"C.7 failed: |r_orig - r_NM| = {result['max_error']:.3e}, "
        f"analytical bound + MC margin = {result['tolerance']:.3e}"
    )


# ----- Aggregate test -----

def test_all_eight_invariances_pass(witness_params):
    """All eight invariance verifications pass at their specified tolerances.

    This is the consolidated §E.5 verification claim: the eight observables of
    the §4 battery satisfy their respective invariance properties (exact or
    near, as specified) when computed on the witness substrate.
    """
    from invariance import verify_all
    results = verify_all(**witness_params)
    failed = [(k, v) for k, v in results.items() if not v["passed"]]
    assert not failed, (
        f"Failed invariance verifications: "
        f"{[(k, v['max_error'], v['tolerance']) for k, v in failed]}"
    )
