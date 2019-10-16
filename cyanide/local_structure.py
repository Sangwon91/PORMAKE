import numpy as np

import ase
import ase.visualize

from .utils import write_molecule_cif

class LocalStructure:
    def __init__(self, positions, indices):
        """
        Local structure of the given position.
        Indices is the indices in the original structure.
        The order of indices is same as  positions.
        The center of local structure is zero vector.
        """
        # Normalize before using.
        self.atoms = ase.Atoms(
                         positions=self.normalize_positions(positions))
        self.indices = np.array(indices, dtype=np.int32)

    @property
    def positions(self):
        return self.atoms.positions

    def normalize_positions(self, positions):
        # Calculate centroid.
        centroid = np.mean(positions, axis=0)

        # Calculate norms of the connection points.
        positions = positions - centroid
        distances = np.linalg.norm(positions, axis=1)

        # Normalize norm of connection points.
        positions = positions / distances[:, np.newaxis]

        # Warning: the centroid of positions are not the zero.
        return positions

    def write_cif(self, filename):
        atoms = ase.Atoms("He") + self.atoms
        bonds = [(0, i) for i in range(len(atoms))]
        bond_types = ["S" for _ in bonds]

        write_molecule_cif(filename, atoms, bonds, bond_types)

    def view(self, show_origin=True):
        if show_origin:
            atoms = self.atoms + ase.Atom("He")
        else:
            atoms = self.atoms
        ase.visualize.view(atoms)
