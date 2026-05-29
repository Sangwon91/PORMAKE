"""The geometric "currency" matched during placement.

This module defines :class:`LocalStructure`, the normalised set of unit
direction vectors around a centroid. Both a topology slot (its directions
to neighbours) and a building block (its directions to connection points)
are expressed as local structures so the :class:`~pormake.locator.Locator`
can align one onto the other independently of size or absolute position.
"""
import ase
import ase.visualize
import numpy as np

from .utils import write_molecule_cif


class LocalStructure:
    """Describe a slot as a set of unit direction vectors.

    A local structure is the normalized geometric "currency" PORMAKE
    uses to compare a topology slot with a building block. Given a set
    of positions (the connection points of a building block, or the
    neighbour directions around a topology node), it strips away
    absolute location and bond length, keeping only the unit direction
    vectors that point from the centroid toward each position. Reducing
    every slot to this common, scale-free form is what lets the
    ``Locator`` search for the rotation and permutation that aligns a
    building block onto a topology slot with minimal RMSD.

    Note that the stored direction vectors are unit-normalized
    individually, so the centroid of the resulting set is generally
    *not* the zero vector even though the original geometry was
    centered first.

    Parameters
    ----------
    positions : array_like of shape (n, 3)
        Cartesian positions defining the slot (e.g. building-block
        connection points or topology neighbour positions).
    indices : array_like of int
        Indices of these positions in the originating structure, kept
        in the same order as ``positions`` so a match found on the
        normalized vectors can be mapped back to concrete atoms.
    normalization_func : callable, optional
        Alternative normalization to apply instead of the default
        unit-direction scheme. When ``None`` the default
        ``normalize_positions`` is used.

    Attributes
    ----------
    atoms : ase.Atoms
        Massless atoms whose positions are the normalized direction
        vectors.
    indices : numpy.ndarray of int32
        The original indices, aligned with ``atoms``.
    """

    def __init__(self, positions, indices, normalization_func=None):
        """Build a local structure from a set of positions."""
        # Normalize before using.
        if normalization_func is not None:
            positions = normalization_func(positions)
        else:
            positions = self.normalize_positions(positions)

        self.atoms = ase.Atoms(positions=positions)
        self.indices = np.array(indices, dtype=np.int32)

    @property
    def positions(self):
        """Normalized direction vectors of the slot.

        These are the values the ``Locator`` rotates and permutes when
        matching this local structure against another.

        Returns
        -------
        numpy.ndarray of shape (n, 3)
            The unit direction vectors stored in ``atoms``.
        """
        return self.atoms.positions

    def normalize_positions(self, positions):
        """Convert positions to unit directions from their centroid.

        Centers the positions on their centroid and rescales each one
        to unit length, yielding the orientation-only description the
        ``Locator`` compares. Because every vector is normalized
        independently, the centroid of the returned set is generally
        not zero.

        Parameters
        ----------
        positions : array_like of shape (n, 3)
            Cartesian positions to normalize.

        Returns
        -------
        numpy.ndarray of shape (n, 3)
            Unit direction vectors pointing from the original centroid
            toward each input position.
        """
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
        """Write the local structure to a CIF file.

        Prepends a helium atom at the origin to mark the centroid and
        draws a bond from it to every direction vector, so the slot's
        geometry can be inspected visually.

        Parameters
        ----------
        filename : str
            Destination path for the CIF file.
        """
        atoms = ase.Atoms("He") + self.atoms
        bonds = [(0, i) for i in range(len(atoms))]
        bond_types = ["S" for _ in bonds]

        write_molecule_cif(filename, atoms, bonds, bond_types)

    def view(self, show_origin=True):
        """Open the local structure in an interactive ASE viewer.

        Convenience helper for visually checking the direction vectors
        of a slot.

        Parameters
        ----------
        show_origin : bool, optional
            If ``True`` (default), add a helium atom at the origin to
            mark the centroid for reference.
        """
        if show_origin:
            atoms = self.atoms + ase.Atom("He")
        else:
            atoms = self.atoms
        ase.visualize.view(atoms)
