"""Align building blocks onto topology slots by rotation and matching.

This module supplies the geometric matching layer of the PORMAKE
pipeline. A topology slot exposes a ``LocalStructure`` (a set of
normalized direction vectors pointing toward the slot's neighbors); a
building block exposes its own ``LocalStructure`` built from the
directions of its connection-point ("X") atoms. To place a building
block in a slot we must decide (a) which connection point maps to which
neighbor direction (a permutation) and (b) the rigid rotation that best
overlays them. Both are solved here so that the assembled framework has
chemically sensible, low-strain geometry.

The two module-level functions solve the sub-problems independently --
optimal assignment via the Hungarian algorithm and optimal rotation via
the Kabsch algorithm -- and the :class:`Locator` class combines them by
brute-forcing trial orientations (Euler-angle grid) to escape the local
optima that a single permutation guess can fall into.
"""

import itertools

import numpy as np
import scipy

from .log import logger


def find_best_permutation(p, q):
    """Find the assignment of ``q`` points onto ``p`` points.

    Solves the linear assignment problem (Hungarian algorithm) on the
    pairwise distance matrix so that each point in ``q`` is matched to a
    distinct point in ``p`` with minimal total distance. This decides
    which connection point of a building block serves which neighbor
    direction of the target slot before a rotation is fitted.

    Parameters
    ----------
    p : numpy.ndarray
        Target points, shape ``(n, 3)`` (e.g. slot neighbor directions).
    q : numpy.ndarray
        Points to be matched against ``p``, shape ``(n, 3)`` (e.g. the
        building block's connection-point directions).

    Returns
    -------
    numpy.ndarray
        Permutation index array of length ``n`` such that ``q[perm]`` is
        aligned, element-wise, with ``p``.
    """
    dist = p[:, np.newaxis] - q[np.newaxis, :]
    dist = np.linalg.norm(dist, axis=-1)

    _, perm = scipy.optimize.linear_sum_assignment(dist)

    return perm


def find_best_orientation(p, q):
    """Find the rotation that best overlays ``q`` onto ``p``.

    Wraps :func:`scipy.spatial.transform.Rotation.align_vectors`
    (Kabsch algorithm) to obtain the rigid rotation minimizing the
    squared distance between the already-paired point sets. The returned
    error is converted from the root-"sum"-squared distance that SciPy
    reports into a per-point root-mean-square deviation (RMSD), the
    quantity the rest of the pipeline uses to compare fits.

    Parameters
    ----------
    p : numpy.ndarray
        Target points, shape ``(n, 3)``; assumed already permuted to
        correspond element-wise with ``q``.
    q : numpy.ndarray
        Points to be rotated onto ``p``, shape ``(n, 3)``.

    Returns
    -------
    U : numpy.ndarray
        ``(3, 3)`` rotation matrix. It is transposed so that it can be
        applied as ``positions @ U`` (right-multiplication of row
        vectors) by the callers.
    rmsd : float
        Per-point RMSD of the aligned configuration.
    """
    # This function gives root "sum" squared distance...
    U, rmsd = scipy.spatial.transform.Rotation.align_vectors(p, q)
    return U.as_matrix().T, np.sqrt(np.square(rmsd) / len(p))


class Locator:
    """Orient a building block to a target slot's local structure.

    A topology slot only fixes the *directions* toward its neighbors;
    the building block must be rotated so that its connection points
    point along those directions. Because the correct point-to-direction
    pairing is not known in advance, this class scans a grid of trial
    orientations over Euler angles, solving the assignment problem at
    each trial and keeping the rotation/permutation pair with the lowest
    RMSD. The Euler scan makes the result robust against the local
    minima that arise when an arbitrary initial permutation is fitted
    directly. The matched building block, its winning permutation, and
    the RMSD are returned for use by :class:`~pormake.builder.Builder`
    and :class:`~pormake.scaler.Scaler`.
    """

    def locate(self, target, bb, max_n_slices=4):
        """Orient ``bb`` onto ``target`` and find the best permutation.

        Scans a coarse Euler-angle grid (whose density adapts to the
        number of connection points and to ``max_n_slices``); at each
        trial orientation it solves for the best connection-point
        permutation, fits the optimal rotation for that permutation, and
        retains the lowest-RMSD result. Only orientation is applied
        here -- translation onto the final cell position happens later
        in the build, after the topology is scaled.

        Parameters
        ----------
        target : LocalStructure
            The slot's local structure (neighbor direction vectors) the
            building block should be aligned to.
        bb : BuildingBlock
            The building block to orient. It is copied before rotation,
            so the input is left unchanged.
        max_n_slices : int, optional
            Controls the Euler-angle grid resolution. The effective
            number of slices per angle is reduced for slots with few
            connection points (2, 3, or 4) and equals ``max_n_slices``
            otherwise. Larger values trade speed for accuracy.

        Returns
        -------
        bb : BuildingBlock
            A rotated copy of the input building block.
        min_perm : numpy.ndarray
            The connection-point permutation that achieved the best fit.
        rmsd_val : float
            Per-point RMSD of the returned orientation.
        """
        local0 = target
        local1 = bb.local_structure()

        # p: target points, q: to be rotated points.
        p_coord = local0.atoms.positions

        # Serching best orientation over Euler angles.
        n_points = p_coord.shape[0]
        if n_points == 2:
            n_slices = 1
        elif n_points == 3:
            n_slices = max_n_slices - 2
        elif n_points == 4:
            n_slices = max_n_slices - 1
        else:
            n_slices = max_n_slices

        logger.debug("n_slices: %d", n_slices)

        alpha = np.linspace(0, 360, n_slices)
        beta = np.linspace(0, 180, n_slices)
        gamma = np.linspace(0, 360, n_slices)

        min_rmsd_val = 1e30
        for a, b, g in itertools.product(alpha, beta, gamma):
            # Copy atoms object for euler rotation.
            atoms = local1.atoms.copy()
            # Rotate.
            atoms.euler_rotate(a, b, g, center=(0, 0, 0))

            # Reorder coordinates.
            q_coord = atoms.positions
            q_perm = find_best_permutation(p_coord, q_coord)

            # Use this permutation of the euler angle. But do not used the
            # Rotated atoms in order to get pure U.
            q_coord = local1.atoms.positions[q_perm]

            # Rotation matrix.
            U, rmsd_val = find_best_orientation(p_coord, q_coord)

            # Save best U and Euler angle.
            if rmsd_val < min_rmsd_val:
                min_rmsd_val = rmsd_val
                min_rmsd_U = U
                min_perm = q_perm

            # The value of 1e-4 can be changed.
            if min_rmsd_val < 1e-4:
                break

        # Load best vals.
        U = min_rmsd_U
        rmsd_val = min_rmsd_val

        # Copy for ratation.
        bb = bb.copy()

        # Rotate using U from RMSD.
        positions = bb.atoms.positions
        centroid = bb.centroid

        positions -= centroid
        positions = np.dot(positions, U) + centroid

        # Update position of atoms.
        bb.atoms.set_positions(positions)

        return bb, min_perm, rmsd_val

    def locate_with_permutation(self, target, bb, permutation):
        """Orient ``bb`` onto ``target`` using a known permutation.

        Skips the expensive Euler-angle search of :meth:`locate` by
        reusing a connection-point permutation that was determined
        earlier (e.g. the one returned by :meth:`locate`, or the trivial
        edge permutation). Only the optimal rotation for that fixed
        permutation is fitted. This is used during the relocation step
        after the topology cell has been scaled, where the pairing is
        already settled and only the new geometry needs re-fitting.

        Parameters
        ----------
        target : LocalStructure
            The slot's local structure to align the building block to.
        bb : BuildingBlock
            The building block to orient; copied before rotation.
        permutation : array_like
            Connection-point permutation mapping ``bb`` points onto the
            target directions.

        Returns
        -------
        bb : BuildingBlock
            A rotated copy of the input building block.
        rmsd_val : float
            Per-point RMSD of the returned orientation.
        """
        local0 = target
        local1 = bb.local_structure()

        # p: target points, q: to be rotated points.
        # Atom types are neglected.
        # p_atoms = np.array(local0.atoms.symbols)
        p_coord = local0.atoms.positions

        # Atom types are neglected.
        # q_atoms = np.array(local1.atoms.symbols)
        q_coord = local1.atoms.positions
        # Permutation used here.
        q_coord = q_coord[permutation]

        # Rotation matrix.
        U, rmsd_val = find_best_orientation(p_coord, q_coord)

        bb = bb.copy()

        # Rotate using U from RMSD.
        positions = bb.atoms.positions
        centroid = bb.centroid

        positions -= centroid
        positions = np.dot(positions, U) + centroid

        # Update position of atoms.
        bb.atoms.set_positions(positions)

        return bb, rmsd_val

    def calculate_rmsd(self, target, bb, max_n_slices=6):
        """Return only the best-fit RMSD of ``bb`` against ``target``.

        Convenience wrapper around :meth:`locate` that discards the
        rotated building block and permutation, keeping just the RMSD.
        The builder uses it to establish the minimum achievable RMSD for
        a given (node type, building block) pair so it can detect when a
        placement is sub-optimal and warrants a higher-accuracy or
        chiral retry.

        Parameters
        ----------
        target : LocalStructure
            The slot's local structure to evaluate the fit against.
        bb : BuildingBlock
            The building block being tested.
        max_n_slices : int, optional
            Euler-angle grid resolution passed through to
            :meth:`locate`; defaults to a finer grid than ``locate`` so
            the reference minimum is reliable.

        Returns
        -------
        float
            Per-point RMSD of the best orientation found.
        """
        _, _, rmsd_val = self.locate(target, bb, max_n_slices)
        return rmsd_val
