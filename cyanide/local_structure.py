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
        """
        Normalize the distance between "center of positions" and
        "a position" to one.
        And move the center of positions to zero
        """
        # Get the center of position.
        cop = np.mean(positions, axis=0)

        # Move cop to zero
        positions = positions - cop

        # Calculate distances for the normalization.
        distances = np.linalg.norm(positions, axis=1)

        # Normalize.
        positions = positions / distances[:, np.newaxis]

        # Get the center of position.
        max_loop = 100
        n_loop = 0
        cop = np.mean(positions, axis=0)
        while (np.abs(cop) > 1e-4).any():
            #print("COP:", cop)
            # Move cop to zero
            positions = positions - cop

            # Calculate distances for the normalization.
            distances = np.linalg.norm(positions, axis=1)

            # Normalize.
            positions = positions / distances[:, np.newaxis]

            # Recenter
            cop = np.mean(positions, axis=0)
            positions = positions - cop

            n_loop += 1
            if n_loop > max_loop:
                logger.warning(
                    f"Max iter in position normalization exceed, Centroid: {cop}"
                )
                break

        # Recenter
        cop = np.mean(positions, axis=0)
        positions = positions - cop

        return positions

    def view(self):
        ase.visualize.view(self.atoms)
