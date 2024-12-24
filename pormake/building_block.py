import copy

import ase
import ase.visualize
import numpy as np

try:
    from ase.utils import natural_cutoffs
except Exception:
    from ase.neighborlist import natural_cutoffs

from .local_structure import LocalStructure
from .log import logger
from .utils import (  # covalent_neighbor_list,
    METAL_LIKE,
    read_budiling_block_xyz,
    write_molecule_cif,
)


class BuildingBlock:
    def __init__(self, bb_file, local_structure_func=None):
        self.atoms = read_budiling_block_xyz(bb_file)
        self.name = self.atoms.info["name"]
        self.connection_point_indices = np.array(self.atoms.info["cpi"])
        self._bonds = self.atoms.info["bonds"]
        self._bond_types = self.atoms.info["bond_types"]
        self._has_metal = None
        self.local_structure_func = local_structure_func

        self.check_bonds()

    def copy(self):
        return copy.deepcopy(self)

    def make_chiral_building_block(self):
        _bb = self.copy()
        _bb.atoms.set_positions(-_bb.atoms.get_positions())
        return _bb

    def local_structure(self):
        connection_points = self.atoms[self.connection_point_indices].positions
        return LocalStructure(
            connection_points,
            self.connection_point_indices,
            normalization_func=self.local_structure_func,
        )

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
        if self._has_metal is None:
            inter = set(self.atoms.symbols) & set(METAL_LIKE)
            return len(inter) != 0
        else:
            logger.debug("has_metal returns self._has_metal.")
            return self._has_metal

    @has_metal.setter
    def has_metal(self, v):
        if isinstance(v, bool) or (v is None):
            logger.debug("New value is assigned for self._has_metal.")
            self._has_metal = v
        else:
            raise Exception("Invalid value for has_metal.")

    @property
    def is_edge(self):
        return self.n_connection_points == 2

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
        # X_indices = set([a.index for a in self.atoms if a.symbol == "X"])
        atom_indices = set([a.index for a in self.atoms])

        sub = list(atom_indices - indices)

        if len(sub) != 0:
            pair = [(i, self.atoms.symbols[i]) for i in sub]
            logger.warning(
                "There are atoms without bond: %s, %s.",
                self.name,
                pair,
            )
            # logger.warning("Make new bond for X.")

    def calculate_bonds(self):
        logger.debug("Start calculating bonds.")

        r = self.atoms.positions
        c = 1.2 * np.array(natural_cutoffs(self.atoms))

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

    def view(self, *args, **kwargs):
        ase.visualize.view(self.atoms, *args, **kwargs)

    def __repr__(self):
        msg = "BuildingBlock: {}, # of connection points: {}".format(
            self.name, self.n_connection_points
        )
        return msg

    def write_cif(self, filename):
        write_molecule_cif(filename, self.atoms, self.bonds, self.bond_types)

    def find_furthest_atom_index(self):
        # Read the XYZ file
        edge = self
        atoms = self.atoms

        # Extract symbols and positions
        positions = atoms.get_positions()

        # Find the indices of the two atoms with the given symbol
        indices = edge.connection_point_indices
        if len(indices) != 2:
            raise ValueError("Number of edge connection point is not 2")

        # Get the coordinates of the two atoms
        coord1 = positions[indices[0]]
        coord2 = positions[indices[1]]

        # Define the axis vector
        axis_vector = coord2 - coord1
        axis_unit_vector = axis_vector / np.linalg.norm(axis_vector)

        # Calculate the projection of each atom onto the axis
        projections = np.dot(positions - coord1, axis_unit_vector)

        # Find the index of the atom with the maximum distance from the axis
        distances = np.linalg.norm(
            (positions - coord1) - np.outer(projections, axis_unit_vector),
            axis=1,
        )
        furthest_atom_index = np.argmax(distances)

        return furthest_atom_index
