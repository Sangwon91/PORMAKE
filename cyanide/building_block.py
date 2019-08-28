import copy

import numpy as np

import ase
import ase.visualize

try:
    from ase.utils import natural_cutoffs
except Exception as e:
    from ase.neighborlist import natural_cutoffs

from .log import logger
from .utils import read_budiling_block_xyz, covalent_neighbor_list, METAL_LIKE
from .local_structure import LocalStructure

class BuildingBlock:
    def __init__(self, bb_file):
        self.atoms = read_budiling_block_xyz(bb_file)
        self.name = self.atoms.info["name"]
        self.connection_point_indices = np.array(self.atoms.info["cpi"])
        self._bonds = self.atoms.info["bonds"]
        self._bond_types = self.atoms.info["bond_types"]

        self.check_bonds()

    def copy(self):
        return copy.deepcopy(self)

    def local_structure(self):
        connection_points = self.atoms[self.connection_point_indices].positions
        return LocalStructure(connection_points, self.connection_point_indices)

    def set_centroid(self, centroid):
        """
        Set centroid of connection points.
        """
        positions = self.atoms.positions
        # Move centroid to zero.
        positions = positions - self.centroid
        # Recentroid by given value.
        positions = positions + centroid
        self.atoms.set_positions(positions)

    @property
    def centroid(self):
        centroid = np.mean(self.connection_points, axis=0)
        return centroid

    @property
    def connection_points(self):
        return self.atoms[self.connection_point_indices].positions

    @property
    def n_connection_points(self):
        return len(self.connection_point_indices)

    @property
    def lengths(self):
        dists = self.connection_points - self.centroid
        norms = np.linalg.norm(dists, axis=1)
        return norms

    @property
    def has_metal(self):
        inter = set(self.atoms.symbols) & set(METAL_LIKE)
        return len(inter) != 0

    @property
    def is_edge(self):
        return self.n_connection_potins == 2

    @property
    def is_node(self):
        return not self.is_edge

    @property
    def bonds(self):
        if self._bonds is None:
            self.calculate_bonds()

        return self._bonds

    @property
    def bond_types(self):
        if self._bond_types is None:
            self.calculate_bonds()

        return self._bond_types

    @property
    def n_atoms(self):
        return len(self.atoms)

    def check_bonds(self):
        if self._bonds is None:
            self.calculate_bonds()

        # Check whether all atoms has bond or not.
        indices = set(np.array(self._bonds).reshape(-1))
        #X_indices = set([a.index for a in self.atoms if a.symbol == "X"])
        atom_indices = set([a.index for a in self.atoms])

        sub = list(atom_indices - indices)

        if len(sub) != 0:
            pair = [(i, self.atoms.symbols[i]) for i in sub]
            logger.warning(
                "There are atoms without bond: %s, %s.", self.name, pair,
            )
            #logger.warning("Make new bond for X.")

    def calculate_bonds(self):
        logger.debug("Start calculating bonds.")

        r = self.atoms.positions
        c = 1.2*np.array(natural_cutoffs(self.atoms))

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
        self._bond_types = ["S" for _ in self.bonds]

    def view(self):
        ase.visualize.view(self.atoms)

    def __repr__(self):
        msg = "BuildingBlock: {}, # of connection points: {}".format(
            self.name, self.n_connection_points
        )
        return msg
