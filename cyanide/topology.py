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

    def copy(self):
        return copy.deepcopy(self)

    def local_structure(self, i):
        indices = []
        positions = []
        for n in self.neighbor_list[i]:
            indices.append(n.index)
            positions.append(n.distance_vector)

        return LocalStructure(positions, indices)

    def get_node_type(self, i):
        return self.atoms.get_tags()[i]

    @property
    def n_all_points(self):
        return len(self.atoms)

    @property
    def node_indices(self):
        return np.argwhere(self.atoms.get_tags() != -1).reshape(-1)

    @property
    def edge_indices(self):
        return np.argwhere(self.atoms.get_tags() == -1).reshape(-1)

    @property
    def n_nodes(self):
        return len(self.node_indices)

    @property
    def n_edges(self):
        return len(self.edge_indices)

    @property
    def n_node_types(self):
        return len(set(self.atoms.get_tags())) - 1

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
