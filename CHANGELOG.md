# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-25

Initial release accompanying the submission of "Persistence Without Individuation:
A Witness Model for Metric-Preserving Identity Underdetermination" (Thomas, 2026).

### Added

- **Witness substrate** (`witness/`): two-oscillator Langevin system with
  exact-OU sampling (bias-free for linear stochastic systems), the orthogonal
  transformation T relating the community partition P₁ to the normal-mode
  partition P₂, and stationary covariance computed both analytically (via the
  continuous-time Lyapunov equation) and numerically (via long-time exact-OU
  simulation).
- **Eight-item observable battery** (`battery/`): persistence autocorrelation,
  return statistics, spectral stability, transport profile, phase-locking
  value (PLV), Kuramoto order parameter, correlation-matrix coherence, and
  broadcast-access (controllability and observability Gramians).
- **Invariance verification scripts** (`invariance/`): one `verify_CN.py` per
  Appendix C section, covering all eight observables. Six observables are
  verified exactly invariant at machine precision; two (PLV and Kuramoto)
  are verified near-invariant within their analytically characterized bounds.
- **Test suite** (`tests/`): 19 fast tests covering witness construction,
  battery computation, and invariance verification, plus 1 slow-marked
  benchmark for the §E.5 1.5%-tolerance stationary-covariance comparison at
  5×10⁶ samples.
- **Demonstration notebook** (`notebooks/fig01_witness.ipynb`): end-to-end
  walkthrough from witness construction through battery computation to
  invariance verification, with figures.
- **Documentation**: `README.md` with installation, quickstart, code-to-paper
  map, repository structure, reproducibility notes, citation block, and
  extension pointers.
- **Licensing**: MIT License for code (`LICENSE`); CC BY 4.0 for documentation
  (`LICENSE-docs`).
- **Release metadata**: `CITATION.cff` for citation tools, `.zenodo.json` for
  the Zenodo DOI deposit, this changelog for version history.

### Verification status

All 19 fast tests pass on Python 3.10–3.12. All eight invariance verifications
pass at their specified tolerances:

| ID  | Observable                  | Max error | Tolerance        |
|-----|-----------------------------|-----------|------------------|
| C.2 | Persistence autocorrelation | 6.66e-16  | 1e-10            |
| C.3 | Return statistics           | 1.58e-2   | 2e-2 (MC)        |
| C.4 | Spectral stability          | 1.33e-15  | 1e-10            |
| C.5 | Transport profile           | 5.55e-17  | 1e-10            |
| C.6 | PLV (near-invariance)       | 0.346     | 0.51 (bound + MC)|
| C.7 | Kuramoto (near-invariance)  | 8.79e-4   | 0.23 (bound + MC)|
| C.8 | Coherence invariants        | 4.44e-16  | 1e-10            |
| C.9 | Broadcast/Gramian           | 2.05e-12  | 1e-10            |

[0.1.0]: https://github.com/EpistriaCST/persistence-without-individuation/releases/tag/v0.1.0
