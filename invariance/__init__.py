"""Invariance verification: numerical confirmation of the §4.2 / Appendix C claims.

Each verify_CN.py script checks the invariance (or characterized near-invariance)
of one battery observable under the orthogonal transformation T relating the
two partitions of the witness substrate.

The eight scripts mirror Appendix C:
  C.2 - Persistence autocorrelation (verify_C2.py)
  C.3 - Return statistics (verify_C3.py)
  C.4 - Spectral stability (verify_C4.py)
  C.5 - Transport profile (verify_C5.py)
  C.6 - Phase-locking value (verify_C6.py)
  C.7 - Kuramoto order parameter (verify_C7.py)
  C.8 - Correlation-matrix coherence (verify_C8.py)
  C.9 - Broadcast-access (verify_C9.py)

Each script can be invoked directly (``python -m invariance.verify_C4``) and
prints a verification report. They are also exercised by the test suite.
"""

from invariance.verify_C2 import verify_C2
from invariance.verify_C3 import verify_C3
from invariance.verify_C4 import verify_C4
from invariance.verify_C5 import verify_C5
from invariance.verify_C6 import verify_C6
from invariance.verify_C7 import verify_C7
from invariance.verify_C8 import verify_C8
from invariance.verify_C9 import verify_C9


def verify_all(omega_0: float = 1.0, k: float = 0.3, gamma: float = 0.1, beta: float = 1.0) -> dict:
    """Run all eight invariance verifications and return a summary dict."""
    return {
        "C2": verify_C2(omega_0, k, gamma, beta),
        "C3": verify_C3(omega_0, k, gamma, beta),
        "C4": verify_C4(omega_0, k, gamma, beta),
        "C5": verify_C5(omega_0, k, gamma, beta),
        "C6": verify_C6(omega_0, k, gamma, beta),
        "C7": verify_C7(omega_0, k, gamma, beta),
        "C8": verify_C8(omega_0, k, gamma, beta),
        "C9": verify_C9(omega_0, k, gamma, beta),
    }


__all__ = [
    "verify_C2", "verify_C3", "verify_C4", "verify_C5",
    "verify_C6", "verify_C7", "verify_C8", "verify_C9",
    "verify_all",
]
