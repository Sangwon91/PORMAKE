"""Molecular building blocks placed onto topology slots.

This module defines :class:`BuildingBlock`, the chemical "piece" that
PORMAKE positions at the nodes and edges of an abstract net. Each block
carries connection points (``X`` atoms) that mark where it bonds to its
neighbours; the :class:`~pormake.locator.Locator` aligns those points to
a slot's local structure and the :class:`~pormake.builder.Builder` fuses
the placed blocks into a periodic framework.
"""
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
    """Represent a molecular fragment placed onto a topology slot.

    A building block is one of the two raw ingredients PORMAKE
    assembles into a framework (the other being the abstract
    ``Topology``). It is read from an ``.xyz`` file in which a few
    atoms carry the special symbol ``"X"``; these *connection points*
    mark where the fragment bonds to its neighbours. The geometric
    arrangement of those connection points around their centroid is
    the building block's ``LocalStructure``, the quantity the
    ``Locator`` rotates and permutes to match a topology slot's local
    structure with minimal RMSD.

    Building blocks come in two flavours determined purely by how many
    connection points they have. A 2-connected fragment is an *edge*
    (a linker that sits on a topology edge-center); anything with three
    or more connection points is a *node* that fills a vertex slot.
    Whether the block contains a metal further distinguishes inorganic
    nodes from organic linkers when reconstructing a MOF.

    Parameters
    ----------
    bb_file : str
        Path to the building block ``.xyz`` file. The file's comment
        line is parsed for the block name, connection-point indices,
        and any pre-recorded bond list / bond types.
    local_structure_func : callable, optional
        Custom normalization applied to the connection points when
        building the ``LocalStructure``. When ``None`` the default
        unit-direction normalization of ``LocalStructure`` is used.
        Supplying a different function lets callers experiment with
        alternative ways of describing slot geometry.

    Attributes
    ----------
    atoms : ase.Atoms
        All atoms of the fragment, including the ``"X"`` connection
        points.
    name : str
        Identifier of the building block, taken from the ``.xyz``
        comment line.
    connection_point_indices : numpy.ndarray
        Indices into ``atoms`` of the ``"X"`` connection points.
    local_structure_func : callable or None
        The normalization function forwarded to ``LocalStructure``.
    """

    def __init__(self, bb_file, local_structure_func=None):
        """Read a building block from an ``.xyz`` file."""
        self.atoms = read_budiling_block_xyz(bb_file)
        self.name = self.atoms.info["name"]
        self.connection_point_indices = np.array(self.atoms.info["cpi"])
        self._bonds = self.atoms.info["bonds"]
        self._bond_types = self.atoms.info["bond_types"]
        self._has_metal = None
        self.local_structure_func = local_structure_func

        self.check_bonds()

    def copy(self):
        """Return an independent deep copy of this building block.

        Used whenever the assembly pipeline needs to place several
        instances of the same fragment onto different slots without
        the per-slot rotations/translations leaking back into the
        shared template.

        Returns
        -------
        BuildingBlock
            A deep copy whose ``atoms`` can be mutated freely.
        """
        return copy.deepcopy(self)

    def make_chiral_building_block(self):
        """Return the mirror image of this building block.

        Inverts every atomic position through the origin, producing
        the opposite-handed enantiomer. This gives the ``Locator`` a
        second candidate to try when a fragment's connection points
        cannot be aligned onto a slot by rotation alone (some slots
        require a reflected building block to reach a low RMSD).

        Returns
        -------
        BuildingBlock
            A copy with all positions negated.
        """
        _bb = self.copy()
        _bb.atoms.set_positions(-_bb.atoms.get_positions())
        return _bb

    def local_structure(self):
        """Build the ``LocalStructure`` of the connection points.

        Wraps the connection-point positions in a ``LocalStructure``,
        i.e. the normalized set of unit direction vectors from their
        centroid. This is the geometric "currency" the ``Locator``
        aligns against a topology slot's local structure to decide
        how the fragment should be oriented.

        Returns
        -------
        LocalStructure
            Local structure of the ``"X"`` connection points, carrying
            their original indices and using
            ``local_structure_func`` for normalization when provided.
        """
        connection_points = self.atoms[self.connection_point_indices].positions
        return LocalStructure(
            connection_points,
            self.connection_point_indices,
            normalization_func=self.local_structure_func,
        )

    def set_centroid(self, centroid):
        """Rigidly translate the fragment to a new centroid.

        Shifts every atom so that the centroid of the connection
        points lands exactly on ``centroid``, leaving the internal
        geometry and orientation untouched. The ``Builder`` calls this
        to drop an already-oriented building block onto the location of
        its target topology slot.

        Parameters
        ----------
        centroid : array_like of shape (3,)
            The Cartesian position the connection-point centroid
            should be moved to.
        """
        positions = self.atoms.positions
        # Move centroid to zero.
        positions = positions - self.centroid
        # Recentroid by given value.
        positions = positions + centroid
        self.atoms.set_positions(positions)

    @property
    def centroid(self):
        """Geometric anchor of the building block.

        The mean of the connection-point positions. This is the point
        the ``Builder`` aligns with a topology slot's location when
        translating a located fragment into place, and the origin
        about which ``Locator`` rotations are applied. It is *not* the
        centroid of all atoms, only of the ``"X"`` connection points.

        Returns
        -------
        numpy.ndarray of shape (3,)
            Mean position of the connection points.
        """
        centroid = np.mean(self.connection_points, axis=0)
        return centroid

    @property
    def connection_points(self):
        """Cartesian positions of the ``"X"`` connection points.

        These are the atoms that bond to neighbouring building blocks;
        their layout defines the slot the fragment can fill and feeds
        both ``centroid`` and ``local_structure``.

        Returns
        -------
        numpy.ndarray of shape (n_connection_points, 3)
            Positions of the connection-point atoms.
        """
        return self.atoms[self.connection_point_indices].positions

    @property
    def n_connection_points(self):
        """Number of connection points (the fragment's coordination).

        This count must equal the coordination number of any topology
        slot the fragment is placed on, and it decides whether the
        fragment is treated as an edge or a node.

        Returns
        -------
        int
            Count of ``"X"`` connection points.
        """
        return len(self.connection_point_indices)

    @property
    def lengths(self):
        """Distances from the centroid to each connection point.

        These radii feed the ``Scaler``, which rescales the topology
        unit cell so node-to-node distances match the real sizes of
        the chosen building blocks.

        Returns
        -------
        numpy.ndarray of shape (n_connection_points,)
            Euclidean distance of each connection point from
            ``centroid``.
        """
        dists = self.connection_points - self.centroid
        norms = np.linalg.norm(dists, axis=1)
        return norms

    @property
    def has_metal(self):
        """Whether the fragment contains a metal-like element.

        Distinguishes inorganic nodes from organic linkers, which the
        pipeline (and the inverse decomposer) needs when classifying
        building blocks. If a value has been explicitly assigned via
        the setter that cached value is returned; otherwise membership
        is computed against ``METAL_LIKE``.

        Returns
        -------
        bool
            ``True`` if any atom symbol is in ``METAL_LIKE`` (or the
            cached override is truthy).
        """
        if self._has_metal is None:
            inter = set(self.atoms.symbols) & set(METAL_LIKE)
            return len(inter) != 0
        else:
            logger.debug("has_metal returns self._has_metal.")
            return self._has_metal

    @has_metal.setter
    def has_metal(self, v):
        """Override the cached metal-presence flag.

        Lets callers force the classification (e.g. when the automatic
        ``METAL_LIKE`` membership test is wrong for an unusual
        fragment). Passing ``None`` clears the override so the
        property falls back to automatic detection.

        Parameters
        ----------
        v : bool or None
            The forced value, or ``None`` to restore auto-detection.

        Raises
        ------
        Exception
            If ``v`` is neither a ``bool`` nor ``None``.
        """
        if isinstance(v, bool) or (v is None):
            logger.debug("New value is assigned for self._has_metal.")
            self._has_metal = v
        else:
            raise Exception("Invalid value for has_metal.")

    @property
    def is_edge(self):
        """Whether this fragment is an edge linker.

        A 2-connected building block is placed on a topology
        edge-center, acting as a linker between two nodes.

        Returns
        -------
        bool
            ``True`` if there are exactly two connection points.
        """
        return self.n_connection_points == 2

    @property
    def is_node(self):
        """Whether this fragment fills a topology node (vertex) slot.

        Any building block that is not an edge (i.e. not 2-connected)
        is treated as a node occupying a vertex of the net.

        Returns
        -------
        bool
            ``True`` if the fragment has other than two connection
            points.
        """
        return not self.is_edge

    @property
    def bonds(self):
        """Atom-index pairs describing the fragment's bonds.

        Used when writing the assembled framework so that bond
        connectivity is preserved in the output CIF. The bond list is
        computed lazily from interatomic distances if it was not
        supplied in the source ``.xyz`` file.

        Returns
        -------
        numpy.ndarray of shape (n_bonds, 2)
            Pairs ``(i, j)`` of bonded atom indices.
        """
        if self._bonds is None:
            self.calculate_bonds()

        return self._bonds

    @property
    def bond_types(self):
        """Bond-order label for each entry in ``bonds``.

        Parallel to ``bonds``; entries are single-bond markers
        (``"S"``) when bonds are auto-computed. Carried through to the
        output CIF alongside the bond list.

        Returns
        -------
        list of str
            One bond-type string per bond.
        """
        if self._bond_types is None:
            self.calculate_bonds()

        return self._bond_types

    @property
    def n_atoms(self):
        """Total number of atoms, including connection points.

        Returns
        -------
        int
            Length of the underlying ``ase.Atoms`` object.
        """
        return len(self.atoms)

    def check_bonds(self):
        """Warn about atoms that participate in no bond.

        Run at construction time as a sanity check: every atom in a
        well-formed fragment should appear in at least one bond. Any
        dangling atoms are logged as a warning (this commonly flags a
        malformed ``.xyz`` or a connection point that failed to bond).
        """
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
        """Infer bonds from interatomic distances.

        Populates ``_bonds`` and ``_bond_types`` when the source file
        carried no explicit connectivity. A pair of atoms is bonded
        when their separation is below 1.2 times the sum of their
        natural covalent cutoffs; every inferred bond is labelled a
        single bond (``"S"``). Each unordered pair is recorded once
        (with ``i < j``).
        """
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
        """Open the fragment in an interactive ASE viewer.

        Convenience wrapper for visual inspection of the atoms (e.g.
        to confirm connection-point placement) during interactive
        work.

        Parameters
        ----------
        *args, **kwargs
            Forwarded verbatim to ``ase.visualize.view``.
        """
        ase.visualize.view(self.atoms, *args, **kwargs)

    def __repr__(self):
        """Return a one-line summary for debugging.

        Returns
        -------
        str
            The building block name and its number of connection
            points.
        """
        msg = "BuildingBlock: {}, # of connection points: {}".format(
            self.name, self.n_connection_points
        )
        return msg

    def write_cif(self, filename):
        """Write the fragment to a CIF file.

        Exports the atoms together with their bond list and bond
        types, so a single building block can be inspected on its own
        outside an assembled framework.

        Parameters
        ----------
        filename : str
            Destination path for the CIF file.
        """
        write_molecule_cif(filename, self.atoms, self.bonds, self.bond_types)
