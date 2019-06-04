import copy

import numpy as np

import ase
import ase.utils
import ase.visualize

from .log import logger
from .utils import read_budiling_block_xyz, covalent_neighbor_list
from .local_structure import LocalStructure

class BuildingBlock:
    def __init__(self, bb_file):
        self.atoms = read_budiling_block_xyz(bb_file)
        self.name = self.atoms.info["name"]
        self.connection_point_indices = np.array(self.atoms.info["cpi"])
        self._bonds = self.atoms.info["bonds"]

    def copy(self):
        return copy.deepcopy(self)

    def local_structure(self):
        connection_points = self.atoms[self.connection_point_indices].positions
        return LocalStructure(connection_points, self.connection_point_indices)

    def set_center(self, center):
        """
        Set center of connection points.
        """
        positions = self.atoms.positions
        # Move center to zero.
        positions = positions - self.center
        # Recenter by given value.
        positions = positions + center
        self.atoms.set_positions(positions)

    @property
    def center(self):
        center = np.mean(self.connection_points, axis=0)
        return center

    @property
    def connection_points(self):
        return self.atoms[self.connection_point_indices].positions

    @property
    def n_connection_points(self):
        return len(self.connection_point_indices)

    @property
    def length(self):
        """
        distance between center and connecting point.
        """
        dists = self.connection_points - self.center
        norm = np.linalg.norm(dists, axis=1)
        # Return average norm. It can be harmful for highly non-symmetric bbs.
        return np.mean(norm)

    @property
    def is_edge(self):
        return self.n_connection_potins == 2

    @property
    def is_node(self):
        return not self.is_edge

    @property
    def bonds(self):
        if self._bonds is None:
            logger.debug("No bonds information. Start bond detection.")
            r = self.atoms.positions
            c = 1.2*np.array(ase.utils.natural_cutoffs(self.atoms))

            diff = r[np.newaxis, :, :] - r[:, np.newaxis, :]
            norms = np.linalg.norm(diff, axis=-1)
            cutoffs = c[np.newaxis, :] + c[:, np.newaxis]

            IJ = np.argwhere(norms < cutoffs)
            I = IJ[:, 0]
            J = IJ[:, 1]

            indices = I < J

            I = I[indices]
            J = J[indices]

            self._bonds = np.stack([I, J], axis=1)
            logger.debug("Bond detection ends.")

        return self._bonds

    @property
    def n_atoms(self):
        return len(self.atoms)

    def view(self):
        ase.visualize.view(self.atoms)

    def __repr__(self):
        msg = "BuildingBlock: {}, # of connection points: {}".format(
            self.name, self.n_connection_points
        )
        return msg
