import itertools

import numpy as np

import sklearn
import sklearn.cluster

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

        p_atoms = np.array(local0.atoms.symbols)
        p_coord = local0.atoms.positions

        q_atoms = np.array(local1.atoms.symbols)
        q_coord = local1.atoms.positions

        q_review = rmsd.reorder_hungarian(p_atoms, q_atoms, p_coord, q_coord)

        q_atoms = q_atoms[q_review]
        q_coord = q_coord[q_review]

        U = rmsd.kabsch(q_coord, p_coord)

        bb = bb.copy()
        bb.atoms.set_positions(bb.atoms.positions @ U)

        rmsd_val = rmsd.kabsch_rmsd(p_coord, q_coord)

        return bb, rmsd_val
