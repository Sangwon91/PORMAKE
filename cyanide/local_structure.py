import numpy as np

import ase
import ase.visualize

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

    def view(self, show_origin=True):
        if show_origin:
            atoms = self.atoms + ase.Atom("He")
        else:
            atoms = self.atoms
        ase.visualize.view(atoms)
