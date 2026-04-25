"""Substrate partitions for the witness system.

The witness state space is R^4 with state vector x = (q1, q2, p1, p2). Two
partitions of the substrate's degrees of freedom into subsystems are considered:

* P1 (community grouping): S_A = {1} groups (q1, p1); S_B = {2} groups (q2, p2).
* P2 (normal-mode grouping): S_+ = {+} groups (Q+, P+); S_- = {-} groups (Q-, P-).

P2 is obtained from P1 by the orthogonal transformation T (see witness.dynamics).
The two partitions induce the same generator (in similarity-equivalent form) and
the same stationary correlation structure, but they assign different state-space
elements to different subsystems.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from witness.dynamics import transformation_T


@dataclass(frozen=True)
class Partition:
    """A partition of the witness substrate into subsystems.

    Attributes
    ----------
    name : str
        Human-readable label.
    subsystem_indices : tuple of tuple of int
        For each subsystem, the indices into the state vector x that belong to it.
        For P1 = {{1}, {2}} with state x = (q1, q2, p1, p2): ((0, 2), (1, 3)).
        For P2 = {{+}, {-}} with state x' = (Q+, Q-, P+, P-): ((0, 2), (1, 3)).
    basis_label : str
        Identifier for the basis the indices refer to: 'original' or 'normal_mode'.
    """

    name: str
    subsystem_indices: tuple[tuple[int, ...], ...]
    basis_label: str


def P1_community() -> Partition:
    """Return the community-grouping partition P1.

    In the original basis x = (q1, q2, p1, p2), subsystem A is the first oscillator
    (indices 0 and 2 — q1 and p1), and subsystem B is the second oscillator
    (indices 1 and 3 — q2 and p2).
    """
    return Partition(
        name="P1_community",
        subsystem_indices=((0, 2), (1, 3)),
        basis_label="original",
    )


def P2_normal_modes() -> Partition:
    """Return the normal-mode-grouping partition P2.

    In the normal-mode basis x' = (Q+, Q-, P+, P-), subsystem + is the symmetric
    mode (indices 0 and 2 — Q+ and P+), and subsystem - is the antisymmetric mode
    (indices 1 and 3 — Q- and P-).
    """
    return Partition(
        name="P2_normal_modes",
        subsystem_indices=((0, 2), (1, 3)),
        basis_label="normal_mode",
    )


def apply_transformation(
    X: NDArray[np.float64],
    T: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply the orthogonal transformation T to a trajectory.

    Parameters
    ----------
    X : ndarray of shape (n, 4)
        Trajectory in the original basis. Each row is a state vector.
    T : ndarray of shape (4, 4), optional
        Orthogonal transformation. Defaults to the canonical T relating P1 to P2.

    Returns
    -------
    X_transformed : ndarray of shape (n, 4)
        Trajectory in the transformed basis. Each row x' satisfies x' = T*x.
    """
    if T is None:
        T = transformation_T()
    return X @ T.T


def project_subsystem(
    X: NDArray[np.float64],
    partition: Partition,
    subsystem: int,
) -> NDArray[np.float64]:
    """Extract the trajectory components belonging to a specific subsystem.

    Parameters
    ----------
    X : ndarray of shape (n, 4)
        Trajectory in the basis matching `partition.basis_label`.
    partition : Partition
        Partition specifying which indices belong to which subsystem.
    subsystem : int
        Index of the subsystem (0 or 1 for the binary partitions of this paper).

    Returns
    -------
    X_sub : ndarray of shape (n, m)
        Subsystem trajectory; columns are the state components in
        `partition.subsystem_indices[subsystem]`.
    """
    indices = partition.subsystem_indices[subsystem]
    return X[:, list(indices)]
