# Persistence Without Individuation — Reference Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Documentation: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-docs)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the reference computational implementation accompanying the
paper *Persistence Without Individuation: A Witness Model for Metric-Preserving
Identity Underdetermination* (Thomas, 2026). It implements the witness substrate
of §3, computes the eight-item observable battery of §4, and numerically verifies
the invariance claims of §4.2 and Appendix C against their analytically specified
tolerances.

The implementation is positioned as **independent verification** of the paper's
analytical results. The core claims of the paper rest on the analytical
treatment in Appendices A–D; the code makes those claims tangible and provides
a substrate on which downstream extensions — including candidate
partition-warrants — can be implemented and tested.

## What the paper argues

Persistence-based observable batteries used by major frameworks of consciousness
(HAOS-IIP, the Free Energy Principle, Global Workspace Theory) are insufficient
to individuate the systems they purport to characterize. On substrates admitting
an admissible transformation group G of dynamical-structure-preserving
transformations, the battery returns identical values across G-related
partitions while the partitions themselves correspond to genuinely distinct
candidate systems. The paper constructs a minimal witness substrate (two
coupled harmonic oscillators with Langevin dissipation) that exhibits this
property explicitly, demonstrates the invariance of eight standard battery
observables both analytically and numerically, and identifies the
*partition-warrant* — a supplementary condition not derivable from the battery
— as the missing element in any framework that claims to individuate.

## What this repository contains

The implementation walks the paper's witness construction end-to-end:

* **The witness substrate** (`witness/`) — the two-oscillator Langevin system
  with the orthogonal transformation T relating the community-grouping
  partition P₁ to the normal-mode-grouping partition P₂. Stationary covariance
  is computed both analytically (via the continuous-time Lyapunov equation)
  and numerically (via long-time exact-OU sampling, a bias-free integrator
  for linear stochastic systems).
* **The eight-item observable battery** (`battery/`) — persistence
  autocorrelation, return statistics, spectral stability, transport profile,
  phase-locking value, Kuramoto order parameter, correlation-matrix coherence,
  and broadcast-access (controllability/observability Gramians).
* **The invariance verification scripts** (`invariance/`) — eight `verify_CN.py`
  scripts mirroring Appendix C, each checking one observable's invariance (or
  characterized near-invariance for PLV and Kuramoto) under the canonical T.
* **A pytest test suite** (`tests/`) — twenty automated checks covering the
  numerical claims of §E.5 of the paper. The suite is split into a fast tier
  (default) and a slow tier (run with `pytest -m slow`) for the
  longer-simulation §E.5 quantitative claim.
* **A demonstration notebook** (`notebooks/fig01_witness.ipynb`) — a worked
  example walking from the witness construction through the battery
  computation to the invariance demonstration, with figures.

## Installation

The implementation requires Python 3.10 or later. Install in editable mode
from the repository root:

```bash
git clone https://github.com/EpistriaCST/persistence-without-individuation.git
cd persistence-without-individuation
pip install -e .
```

Or, with the optional development dependencies for testing and notebooks:

```bash
pip install -e ".[dev,notebooks]"
```

The core runtime dependencies are `numpy >= 1.24`, `scipy >= 1.11`, and
`matplotlib >= 3.7`. The test suite additionally requires `pytest >= 7.4`,
and the notebooks require `jupyter`.

## Quickstart

Verify all eight invariance claims with a single command:

```python
from invariance import verify_all

results = verify_all(omega_0=1.0, k=0.3, gamma=0.1, beta=1.0)
for name, result in results.items():
    print(f"{name}: passed={result['passed']}, max_error={result['max_error']:.3e}")
```

Inspect a specific verification with verbose output:

```python
from invariance import verify_C2

verify_C2(omega_0=1.0, k=0.3, gamma=0.1, beta=1.0, verbose=True)
```

Or run the test suite from the command line:

```bash
pytest                  # fast tier only (~12 seconds)
pytest -m slow          # include the slow §E.5 1%-tolerance benchmark
pytest --cov            # with coverage (requires pytest-cov)
```

## Mapping code to paper claims

Each section of the paper that makes a numerical claim is implemented at a
specific code location:

| Paper section | Claim | Implementation |
| --- | --- | --- |
| §3.2, §B.4 | Witness stationary covariance | `witness/stationary.py` |
| §4.2, §C.2 | Persistence autocorrelation invariance | `invariance/verify_C2.py` |
| §4.2, §C.3 | Return statistics invariance up to region transformation | `invariance/verify_C3.py` |
| §4.2, §C.4 | Spectral-stability invariance | `invariance/verify_C4.py` |
| §4.2, §C.5 | Transport-profile invariance | `invariance/verify_C5.py` |
| §4.2, §C.6 | PLV near-invariance with characterized bound | `invariance/verify_C6.py` |
| §4.2, §C.7 | Kuramoto near-invariance | `invariance/verify_C7.py` |
| §4.2, §C.8 | Correlation-matrix coherence invariance | `invariance/verify_C8.py` |
| §4.2, §C.9 | Broadcast/Gramian invariance | `invariance/verify_C9.py` |
| §6.1 | Identity-set divergence Ind(P₁) ≠ Ind(P₂) | `witness/partitions.py` |
| §E.5 | Test suite asserting numerical claims | `tests/test_invariance.py`, `tests/test_witness.py` |

## Repository structure

```
persistence-without-individuation/
├── README.md
├── LICENSE                  # MIT (code)
├── LICENSE-docs             # CC BY 4.0 (documentation)
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── witness/                 # Two-oscillator Langevin witness
│   ├── __init__.py
│   ├── dynamics.py          # Drift matrix, diffusion matrix, T, exact-OU sampler
│   ├── partitions.py        # P₁ and P₂, transformation application
│   └── stationary.py        # Stationary and lagged covariance
├── battery/                 # Eight-item battery observables
│   ├── __init__.py
│   ├── autocorrelation.py
│   ├── returns.py
│   ├── spectral.py
│   ├── transport.py
│   ├── plv.py
│   ├── kuramoto.py
│   ├── coherence.py
│   └── broadcast.py
├── invariance/              # Verification of Appendix C claims
│   ├── __init__.py
│   ├── verify_C2.py through verify_C9.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_witness.py
│   └── test_invariance.py
└── notebooks/
    └── fig01_witness.ipynb
```

## Reproducibility notes

The exact OU sampler used in `witness.dynamics.simulate` is bias-free for
linear stochastic systems, with no O(dt) bias of the kind incurred by
Euler-Maruyama integration. This is the appropriate integrator for verifying
the paper's claims because the witness is exactly linear: any residual
disagreement between analytical and numerical results is sampling variance
rather than discretization error.

The test suite uses fixed random seeds and reports tolerances explicitly. The
`test_stationary_covariance_analytical_vs_numerical` test runs at 10⁶ samples
with a 3% Frobenius-norm tolerance reflecting the statistical limit at that
sample size; the slow-marked `test_stationary_covariance_E5_claim` runs at
5×10⁶ samples with a 1.5% tolerance for a tighter agreement check.

The PLV and Kuramoto verifications use trajectory-based estimators rather
than analytical expressions because these observables involve the Hilbert
transform, which is most naturally computed on a finite trajectory. The
verifications check the structural-near-invariance bound from §4.2 — the
Frobenius-norm difference of PLV matrices and the absolute difference of
Kuramoto r are bounded by analytical functions of the dimensionless coupling
α = k / (ω₀² + k) and ε = k / ω₀² respectively.

## Citation

To cite this implementation:

```bibtex
@software{thomas_2026_persistence_without_individuation_code,
  author       = {Thomas, Charles S.},
  title        = {Computational implementation for
                  ``Persistence Without Individuation'':
                  Witness construction, battery computation, and
                  invariance verification},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {[to be assigned upon Zenodo release]},
  url          = {https://github.com/EpistriaCST/persistence-without-individuation}
}
```

To cite the paper itself:

```bibtex
@article{thomas_2026_persistence_without_individuation,
  author       = {Thomas, Charles S.},
  title        = {Persistence Without Individuation:
                  A Witness Model for Metric-Preserving Identity Underdetermination},
  year         = 2026,
  journal      = {[venue to be confirmed]},
  note         = {Manuscript in submission}
}
```

## Extensions

The repository is structured to support downstream research. Three classes of
extension are anticipated:

* **Implementing candidate partition-warrants.** The four partition-warrant
  families enumerated in §7.3 — optimization-based, structural-definitional,
  statistical-structural, and process-based — can each be implemented as a
  predicate on the witness, with a common interface returning a selection
  score or pass/fail verdict on each candidate partition.
* **Extending to additional frameworks.** The §5.4 classification criterion
  applies to frameworks beyond the three walked through in §5. Adding a new
  framework requires implementing its observables in `battery/` and adding
  the corresponding invariance-verification script in `invariance/`.
* **Testing substrates beyond the witness.** The core infrastructure
  generalizes to arbitrary linear-Gaussian substrates specified by drift
  matrix A and diffusion matrix D. Users can specify their own systems and
  apply the battery and invariance pipeline without modifying the core code.

## License

* The code in this repository is released under the [MIT License](LICENSE).
* The documentation (this README, the docstrings, and the notebooks) is
  released under [Creative Commons Attribution 4.0 International](LICENSE-docs).

## Related work

The paper extends and applies a structural pattern familiar from non-
identifiability arguments in physics (gauge redundancy), statistics
(sufficient-statistics arguments), network science (community-detection
non-uniqueness), and information theory (partition-dependent information
measures). The contribution is to apply this pattern to the methodology of
individuation in consciousness-adjacent frameworks, and to identify the
partition-warrant slot as the structural locus of any positive
individuation theory. The companion paper (Thomas, forthcoming) takes up
the question of what partition-warrant is correct, proposing closure
activation as the supplementary condition.

## Contact

Charles S. Thomas — Epsitria, LLC — `charles@epistria.com`

Issues, questions, and suggestions are welcome via the
[GitHub issue tracker](https://github.com/EpistriaCST/persistence-without-individuation/issues).
