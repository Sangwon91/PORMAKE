import itertools

import numpy as np

from .log import logger
from .third_party.rmsd import rmsd

class Locator:
    def locate(self, target, bb, max_n_slices=4):
        """
        Locate building block (bb) to target_points
        using the connection points of the bb.

        Return:
            located building block and RMS.
        """
        local0 = target
        local1 = bb.local_structure()

        # p: target points, q: to be rotated points.
        p_atoms = np.array(local0.atoms.symbols)
        p_coord = local0.atoms.positions

        q_atoms = np.array(local1.atoms.symbols)

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
            q_perm = rmsd.reorder_hungarian(
                           p_atoms, q_atoms, p_coord, q_coord)

            # Use this permutation of the euler angle. But do not used the
            # Rotated atoms in order to get pure U.
            q_coord = local1.atoms.positions[q_perm]

            # Rotation matrix.
            U = rmsd.kabsch(q_coord, p_coord)
            rmsd_val = rmsd.kabsch_rmsd(p_coord, q_coord)

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
        """
        Locate bb to target with pre-obtained permutation of bb.
        """
        local0 = target
        local1 = bb.local_structure()

        # p: target points, q: to be rotated points.
        p_atoms = np.array(local0.atoms.symbols)
        p_coord = local0.atoms.positions

        q_atoms = np.array(local1.atoms.symbols)
        q_coord = local1.atoms.positions
        # Permutation used here.
        q_coord = q_coord[permutation]

        # Rotation matrix.
        U = rmsd.kabsch(q_coord, p_coord)
        rmsd_val = rmsd.kabsch_rmsd(p_coord, q_coord)

        bb = bb.copy()

        # Rotate using U from RMSD.
        positions = bb.atoms.positions
        centroid = bb.centroid

        positions -= centroid
        positions = np.dot(positions, U) + centroid

        # Update position of atoms.
        bb.atoms.set_positions(positions)

        return bb, rmsd_val
