"""Connectivity between topology nodes and edge centers.

This module turns the bare atomic positions of a parsed net into an
explicit adjacency structure. :class:`Neighbor` is a single
(index, distance-vector) record and :class:`NeighborList` builds the full
node-edge connectivity used everywhere downstream: the
:class:`~pormake.topology.Topology` derives local structures and edge
types from it, and the :class:`~pormake.scaler.Scaler` rewrites it with
rescaled edge geometry after cell optimization.
"""
import ase
import ase.neighborlist
import numpy as np

from .log import logger


class Neighbor:
    """Hold a single topology connectivity record.

    A ``Neighbor`` is one entry in a slot's neighbor list: it pairs the
    integer index of an adjacent slot (a node or an edge-center in the
    underlying ``ase.Atoms``) with the minimum-image displacement vector
    pointing from the owning slot toward that neighbor. Storing the
    displacement (rather than just the index) keeps periodic-boundary
    information intact, which is what lets ``Topology`` recover each
    slot's local geometry and lets the ``Locator`` align building blocks.

    Attributes
    ----------
    index : int
        Index, in the topology's ``ase.Atoms`` object, of the neighboring
        node or edge-center slot.
    distance_vector : numpy.ndarray
        Minimum-image Cartesian displacement from the owning slot to the
        neighbor; its direction defines part of the slot's local
        structure and its norm is the through-space distance.
    """

    def __init__(self, index, distance_vector):
        """Store the neighbor's slot index and displacement vector."""
        self.index = index
        self.distance_vector = distance_vector

    def __repr__(self):
        """Return a one-line ``index``/``distance vector`` summary."""
        return "index: {}, distance vector: {}".format(
            self.index, self.distance_vector
        )


class NeighborList:
    """Build node-to-edge-center connectivity for a topology.

    A topology net is stored as an ``ase.Atoms`` object in which carbon
    ("C") atoms mark node (vertex) slots and oxygen ("O") atoms mark
    edge-center slots. ``NeighborList`` scans that geometry once and
    records, for every slot, which slots it is bonded to and the
    minimum-image displacement to each. The result is a list indexed by
    slot: ``neighbor_list[i]`` yields the ``Neighbor`` records of slot
    ``i``. Every edge-center should end up with exactly two node
    neighbors, and each node with as many edge-center neighbors as its
    coordination number; ``Topology`` relies on this object to validate
    those counts and to build per-slot ``LocalStructure`` objects.

    Two construction strategies are offered because ``.cgd`` files vary
    in numerical precision: ``"distance"`` bonds C-O pairs within a fixed
    cutoff, while ``"nearest"`` keeps each edge-center's two closest
    nodes and prunes asymmetric bonds. ``Topology`` tries the cheaper
    distance method first and falls back to the nearest-two method when
    validation fails.

    Parameters
    ----------
    atoms : ase.Atoms
        Topology geometry whose "C" atoms are node slots and "O" atoms
        are edge-center slots.
    method : {"distance", "nearest"}
        Connectivity strategy. ``"distance"`` uses a fixed C-O cutoff;
        ``"nearest"`` keeps the two nearest nodes per edge-center and
        removes non-reciprocal bonds.

    Attributes
    ----------
    max_index : int
        Largest slot index seen while building, i.e. one less than the
        number of slots that own a neighbor list.
    _neighbor_list : list of list of Neighbor
        Per-slot adjacency: entry ``i`` holds the ``Neighbor`` records
        for slot ``i``.

    Raises
    ------
    Exception
        If ``method`` is neither ``"distance"`` nor ``"nearest"``.
    """

    def __init__(self, atoms, method):
        """Build connectivity from ``atoms`` using the chosen ``method``."""
        # C for nodes and O for edges.
        if method == "distance":
            self.distance_based_build(atoms)
        elif method == "nearest":
            self.nearest_two_based_build(atoms)
        else:
            logger.error(f"Invalid method {method}.")
            raise Exception("Invalid arguments.")  # Hmm...

    def distance_based_build(self, atoms):
        """Build connectivity by bonding C-O pairs within a fixed cutoff.

        Uses ASE's neighbor search with a small C-O cutoff (and zero C-C
        and O-O cutoffs) so that only node-to-edge-center bonds are kept.
        This is the fast, precision-sensitive default; ``Topology`` calls
        it first and only falls back to ``nearest_two_based_build`` if the
        resulting coordination numbers do not validate.

        Parameters
        ----------
        atoms : ase.Atoms
            Topology geometry whose "C" atoms are nodes and "O" atoms are
            edge-centers.
        """
        eps = 1e-3
        cutoffs = {
            ("C", "C"): 0.0,
            ("O", "O"): 0.0,
            ("C", "O"): 0.5 + eps,
        }

        I, J, D = ase.neighborlist.neighbor_list("ijD", atoms, cutoff=cutoffs)

        self.max_index = np.max(I)
        self._neighbor_list = [[] for _ in range(self.max_index + 1)]

        for i, j, d in zip(I, J, D):
            self._neighbor_list[i].append(Neighbor(j, d))

    def nearest_two_based_build(self, atoms):
        """Build connectivity by keeping each edge-center's two nearest nodes.

        This fallback strategy first gathers C-O bonds within a looser
        cutoff, then for every edge-center ("O") slot retains only the two
        nodes closest to it (an edge always joins exactly two nodes), and
        finally drops any node-side neighbor that does not reciprocate.
        It is more tolerant of imprecise ``.cgd`` coordinates than the
        distance method, so ``Topology`` uses it when distance-based
        parsing fails validation.

        Parameters
        ----------
        atoms : ase.Atoms
            Topology geometry whose "C" atoms are nodes and "O" atoms are
            edge-centers.
        """
        # C for nodes and O for edges.
        cutoffs = {
            ("C", "C"): 0.0,
            ("O", "O"): 0.0,
            ("C", "O"): 0.7,
        }

        I, J, D = ase.neighborlist.neighbor_list("ijD", atoms, cutoff=cutoffs)

        self.max_index = np.max(I)
        self._neighbor_list = [[] for _ in range(self.max_index + 1)]

        for i, j, d in zip(I, J, D):
            self._neighbor_list[i].append(Neighbor(j, d))

        # Pick nearest 2 nodes.
        edge_indices = np.argwhere(atoms.symbols == "O").reshape(-1)
        for i in edge_indices:
            neighbor = self._neighbor_list[i]
            # Pick 2 shortest distances
            neighbor.sort(key=lambda x: np.linalg.norm(x.distance_vector))
            self._neighbor_list[i] = neighbor[:2]

        # Remove invalid neighbors of nodes.
        node_indices = np.argwhere(atoms.symbols == "C").reshape(-1)
        for i in node_indices:
            neighbor = []
            for ni in self._neighbor_list[i]:
                j = ni.index
                # Check cross reference.
                if i in [nj.index for nj in self._neighbor_list[j]]:
                    neighbor.append(ni)
            self._neighbor_list[i] = neighbor

    def __getitem__(self, i):
        """Return the list of ``Neighbor`` records for slot ``i``.

        Parameters
        ----------
        i : int
            Slot index (a node or edge-center) in the topology.

        Returns
        -------
        list of Neighbor
            Connectivity records of slot ``i``.
        """
        return self._neighbor_list[i]

    def __iter__(self):
        """Iterate over slots, yielding each slot's ``Neighbor`` list.

        Returns
        -------
        iterator
            Iterator over the per-slot lists of ``Neighbor`` records, in
            slot-index order.
        """
        return iter(self._neighbor_list)

    def set_data(self, data):
        """Replace the connectivity with externally supplied records.

        Rebuilds the internal per-slot adjacency from raw
        ``(index, distance_vector)`` pairs. ``Scaler`` calls this after
        rescaling the topology cell so the neighbor list reflects the new,
        building-block-sized geometry while preserving which slots connect.

        Parameters
        ----------
        data : iterable of iterable of (int, numpy.ndarray)
            Per-slot connectivity. Each inner iterable holds
            ``(index, distance_vector)`` pairs that become ``Neighbor``
            records for that slot.
        """
        new_list = []
        for neighbor in data:
            new_list.append([])
            for n in neighbor:
                new_list[-1].append(Neighbor(n[0], n[1]))

        self._neighbor_list = new_list

    def __repr__(self):
        """Return a multi-line per-slot listing of neighbor counts and records.

        Returns
        -------
        str
            One block per slot showing ``slot: count`` followed by each
            ``Neighbor`` record, for quick inspection of the connectivity.
        """
        output = ""
        for i, neighbor in enumerate(self):
            line = "{}: {}\n".format(i, len(neighbor))
            for n in neighbor:
                line += "{}\n".format(n)
            output += line
        return output
