import copy

import numpy as np

import ase
import ase.visualize

from .utils import read_cgd
from .local_structure import LocalStructure
from .neighbor_list import Neighbor, NeighborList


class Topology:
    def __init__(self, cgd_file):
        self.atoms = read_cgd(filename=cgd_file)
        self.neighbor_list = NeighborList(self.atoms)
        # Save additional information.
        self.name = self.atoms.info["name"]
        self.spacegroup = self.atoms.info["spacegroup"]

        # Calculate properties.
        self.calculate_properties()

    def copy(self):
        return copy.deepcopy(self)

    def calculate_properties(self):
        # Build indices of nodes and edges.
        self._node_indices = \
            np.argwhere(self.atoms.get_tags() != -1).reshape(-1)
        self._edge_indices = \
            np.argwhere(self.atoms.get_tags() == -1).reshape(-1)

        # Build node type.
        self._node_types = self.atoms.get_tags()

        # Build edge type.
        self._edge_types = [(-1, -1) for _ in range(self.n_all_points)]
        for i in self.edge_indices:
            n = self.neighbor_list[i]

            i0 = n[0].index
            i1 = n[1].index

            t0 = self.get_node_type(i0)
            t1 = self.get_node_type(i1)

            # Sort.
            if t0 > t1:
                t0, t1 = t1, t0

            self._edge_types[i] = (t0, t1)
        self._edge_types = np.array(self._edge_types)

        # Calculate the number of node and edge types.
        self._n_node_types = np.unique(self.node_types).shape[0] - 1
        self._n_edge_types = np.unique(self.edge_types, axis=0).shape[0] - 1

    def local_structure(self, i):
        indices = []
        positions = []
        for n in self.neighbor_list[i]:
            indices.append(n.index)
            positions.append(n.distance_vector)

        return LocalStructure(positions, indices)

    def get_node_type(self, i):
        return self._node_types[i]

    def get_edge_type(self, i):
        return self._edge_types[i]

    @property
    def node_types(self):
        return self._node_types

    @property
    def edge_types(self):
        return self._edge_types

    @property
    def n_all_points(self):
        return len(self.atoms)

    @property
    def node_indices(self):
        return self._node_indices
    @property
    def edge_indices(self):
        return self._edge_indices

    @property
    def n_nodes(self):
        return len(self.node_indices)

    @property
    def n_edges(self):
        return len(self.edge_indices)

    @property
    def n_node_types(self):
        return self._n_node_types

    @property
    def n_edge_types(self):
        return self._n_edge_types

    def get_neigbor_indices(self, i):
        return [n.index for n in self.neighbor_list[i]]

    def get_edge_length(self, i):
        n1, n2 = self.neighbor_list[i]
        diff = n1.distance_vector - n2.distance_vector
        return np.linalg.norm(diff)

    def view(self, show_edge_centers=True, repeat=1, **kwargs):
        atoms = self.atoms.copy()

        if show_edge_centers:
            scale = 3
            # Replace symbol O to F for bond visualization.
            symbols = np.array(atoms.get_chemical_symbols())
            symbols[symbols == "O"] = "F"
            atoms.set_chemical_symbols(symbols)
        else:
            scale = 2
            del atoms[atoms.symbols == "O"]

        # Expand cell for the visualization.
        s = atoms.get_scaled_positions()
        atoms.set_cell(atoms.cell*scale)
        atoms.set_positions(s @ atoms.cell)

        # Visualize.
        r = repeat
        if isinstance(r, int):
            r = (r, r, r)
        ase.visualize.view(atoms, repeat=r, **kwargs)
