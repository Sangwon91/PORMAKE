import numpy as np

import ase
import ase.visualize

from .utils import normalize_positions

class LocalStructure:
    def __init__(self, positions, indices):
        """
        Local structure of the given position.
        Indices is the indices in the original structure.
        The order of indices is same as  positions.
        The center of local structure is zero vector.
        """
        # Normalize before using.
        self.atoms = ase.Atoms(positions=normalize_positions(positions))
        self.indices = np.array(indices, dtype=np.int32)

    @property
    def positions(self):
        return self.atoms.positions

    def matching_permutation(self, other):
        """
        Match one-to-one index to other local structure.
        self.positions = other.positions[matching_permutation]
        """
        pos_i = self.positions
        pos_j = other.positions

        assert len(pos_i) == len(pos_j)

        # Calculate all distances.
        diffs = pos_i[:, np.newaxis, :] - pos_j[np.newaxis, :, :]
        norms = np.linalg.norm(diffs, axis=2)

        # Get minimum distance indices as permutation.
        permutation = np.argmin(norms, axis=1)

        # Check uniqueness.
        assert len(set(permutation)) == len(permutation)

        return permutation

    def view(self):
        ase.visualize.view(self.atoms)
