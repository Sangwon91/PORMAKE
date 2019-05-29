import itertools

import numpy as np

from .third_party.rmsd import rmsd

class Locator:
    def locate(self, target, bb):
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
        alpha = np.linspace(0, 360, 4)
        beta = np.linspace(0, 180, 4)
        gamma = np.linspace(0, 360, 4)

        min_rmsd_val = 1e30
        for a, b, g in itertools.product(alpha, beta, gamma):
            # Copy atoms object for euler rotation.
            atoms = local1.atoms.copy()
            # Rotate.
            atoms.euler_rotate(a, b, g)

            # Reorder coordinates.
            q_coord = atoms.positions
            q_review = rmsd.reorder_hungarian(
                           p_atoms, q_atoms, p_coord, q_coord)
            q_coord = q_coord[q_review]

            # Rotation matrix.
            U = rmsd.kabsch(q_coord, p_coord)
            rmsd_val = rmsd.kabsch_rmsd(p_coord, q_coord)

            # Save best U and Euler angle.
            if rmsd_val < min_rmsd_val:
                min_rmsd_val = rmsd_val
                min_rmsd_U = U
                min_euler_angle = (a, b, g)

            # The value of 1e-4 can be changed.
            if min_rmsd_val < 1e-4:
                break

        # Load best vals.
        U = min_rmsd_U
        rmsd_val = min_rmsd_val
        # Rotate building block.
        bb = bb.copy()
        # Apply euler rotate of minimum RMSD.
        bb.atoms.euler_rotate(*min_euler_angle)

        # Rotate using U from RMSD.
        positions = bb.atoms.positions
        center = bb.center

        positions -= center
        positions = np.dot(positions, U) + center

        # Update position of atoms.
        bb.atoms.set_positions(positions)

        return bb, rmsd_val
