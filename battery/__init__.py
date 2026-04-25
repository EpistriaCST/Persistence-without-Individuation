"""Eight-item observable battery from §4.1 of the paper.

The battery is the inventory of identity-relevant observables that recur across
persistence-based frameworks (HAOS, FEP, GWT). Each observable is implemented
as a function or class that operates on either a trajectory or the stationary
covariance and drift matrix of the substrate.

The eight observables:
  1. Persistence autocorrelation (autocorrelation.py)
  2. Return statistics (returns.py)
  3. Spectral stability (spectral.py)
  4. Transport profile (transport.py)
  5. Phase-locking value (plv.py)
  6. Kuramoto order parameter (kuramoto.py)
  7. Correlation-matrix coherence (coherence.py)
  8. Broadcast-access (Gramian) (broadcast.py)
"""

from battery.autocorrelation import autocorrelation
from battery.returns import return_probability_basis
from battery.spectral import spectrum
from battery.transport import transport_eigenvalues
from battery.plv import phase_locking_value
from battery.kuramoto import kuramoto_order_parameter
from battery.coherence import coherence_invariants
from battery.broadcast import controllability_gramian, observability_gramian, gramian_invariants

__all__ = [
    "autocorrelation",
    "return_probability_basis",
    "spectrum",
    "transport_eigenvalues",
    "phase_locking_value",
    "kuramoto_order_parameter",
    "coherence_invariants",
    "controllability_gramian",
    "observability_gramian",
    "gramian_invariants",
]
