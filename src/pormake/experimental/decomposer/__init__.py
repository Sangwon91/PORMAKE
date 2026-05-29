"""Decompose an existing MOF back into building blocks.

This experimental subpackage runs the build pipeline in reverse: given a
finished MOF crystal it identifies the metal nodes and organic linkers,
cuts the bonds between them, and emits capped fragments that can be reused
as PORMAKE building blocks. It is refactored from the legacy code used in
the PORMAKE paper and may not be stable.
"""
import collections
import copy
from pathlib import Path

import ase
import networkx as nx
import numpy as np

from ...utils import METAL_LIKE, covalent_neighbor_list


def hash_atoms(atoms: ase.Atoms, complexity: int = 6):
    """Hash an ase.Atoms object into a near-unique integer fingerprint.

    The fingerprint is built from the covalent-bond adjacency matrix and
    the atomic numbers, making it invariant to atom ordering. It lets the
    decomposer group chemically identical fragments so that recurring
    building blocks are recognised as the same piece. Experimental and may
    not be stable.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to be hashed.
    complexity : int
        The number of times to apply the adjacency matrix to the feature matrix.

    Returns
    -------
    int
        The hashed integer.
    """
    X = np.array([a.number for a in atoms], dtype=np.int32)[:, np.newaxis]
    X = np.concatenate([X, X**2], axis=1)

    i, j, _ = covalent_neighbor_list(atoms)
    n = len(atoms)
    A = np.eye(n, dtype=np.int32)
    A[i, j] = 1

    H = X
    for _ in range(complexity):
        H = A @ H
    H = np.sum(np.mean(np.sqrt(H), axis=1))

    return int(np.around(H, decimals=2) * 100)


def estimate_atoms_dimension(atoms: ase.Atoms):
    """Estimate the periodic dimensionality (0-3) of an atoms object.

    Used to tell a finite molecular building block (0D) apart from a chain
    (1D), layer (2D), or framework (3D); the current PORMAKE only supports
    0D building blocks, so this guards fragment merging during
    decomposition. Experimental and may not be stable.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to be estimated.

    Returns
    -------
    int
        The estimated dimension.
    """
    I, J, _ = covalent_neighbor_list(atoms)
    graph = nx.Graph(zip(I, J))
    original_dim = len(list(nx.connected_components(graph)))

    # Apply 2 x 2 x 2 expansion.
    I, J, _ = covalent_neighbor_list(atoms * 2)
    graph = nx.Graph(zip(I, J))
    new_dim = len(list(nx.connected_components(graph)))

    ratio = new_dim // original_dim

    vmap = {
        8: 0,
        4: 1,
        2: 2,
        1: 3,
    }

    return vmap[ratio]


def remove_pbc_cuts(atoms):
    """Remove building block cuts due to periodic boundary conditions. After the
    removal, the atoms object is centered at the center of the unit cell.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to be processed.

    Returns
    -------
    ase.Atoms
        The processed atoms object.
    """
    I, J, D = covalent_neighbor_list(atoms)

    nl = [[] for _ in atoms]
    for i, j, d in zip(I, J, D):
        nl[i].append((j, d))

    visited = [False for _ in atoms]
    q = collections.deque()

    # Center of the unit cell.
    abc_half = np.sum(atoms.get_cell(), axis=0) * 0.5

    positions = {}
    q.append((0, np.array([0.0, 0.0, 0.0])))
    while q:
        i, pos = q.pop()
        visited[i] = True
        positions[i] = pos
        for j, d in nl[i]:
            if not visited[j]:
                q.append((j, pos + d))
                visited[j] = True

    centroid = np.array([0.0, 0.0, 0.0])
    for v in positions.values():
        centroid += v
    centroid /= len(positions)

    syms = [None for _ in atoms]
    poss = [None for _ in atoms]
    for i in range(len(atoms)):
        syms[i] = atoms.symbols[i]
        poss[i] = positions[i] - centroid + abc_half

    atoms = ase.Atoms(
        symbols=syms, positions=poss, pbc=True, cell=atoms.get_cell()
    )

    return atoms


class MOFDecomposer:
    """Decompose an existing MOF back into building blocks.

    This is the inverse of the PORMAKE assembly pipeline: instead of
    placing building blocks onto a topology to grow a framework, it
    reads an assembled MOF from a CIF and cuts it apart into the
    inorganic nodes and organic linkers it was built from. The metal
    coordination bonds are treated as the seams between fragments;
    severing them yields the individual building blocks, and the
    severed bonds (the *connecting bonds*) record where each fragment
    reconnected to its neighbours. Those break points are later capped
    with dummy ``X`` atoms so the extracted fragments carry the same
    connection-point convention as native PORMAKE building blocks.

    It is an experimental feature and may not be stable.

    Parameters
    ----------
    cif : str
        The path to the CIF file of the MOF.
    X_type : str
        The symbol used for the dummy connection-point atom appended at
        each break point. It is used to identify the connection sites.
        Default is ``'X'``.

    Attributes
    ----------
    atoms : ase.Atoms
        The MOF structure read from the CIF (mutated in place by
        ``cleanup``).
    name : str
        The CIF file stem, used to name the structure.
    bb_found : bool
        Whether building blocks and connecting bonds have been
        computed yet (drives lazy evaluation).
    X_type : str
        Symbol used for the connection-point dummy atoms.

    TODO:
        * Custom bond information (connectivity and bond types).
    """

    def __init__(self, cif, X_type='X'):
        """Read the MOF from a CIF file."""
        self.atoms = ase.io.read(cif)
        self.name = Path(cif).stem
        self.bb_found = False
        self.X_type = X_type

    def view(self, *args, **kwargs):
        """Open the MOF structure in an interactive ASE viewer.

        Convenience helper for visually inspecting the loaded
        structure, e.g. before or after ``cleanup``.

        Parameters
        ----------
        *args, **kwargs
            Forwarded verbatim to ``ase.visualize.view``.
        """
        ase.visualize.view(self.atoms, *args, **kwargs)

    def cleanup(self, remove_interpenetration=True):
        """Remove interpenetration and isolated molecules from the MOF.

        A clean single framework is a precondition for reliable
        decomposition, so this prunes the structure down to the relevant
        connected component(s) before fragments are searched for.
        Experimental and may not be stable.

        Parameters
        ----------
        remove_interpenetration : bool
            If True, keep only the largest framework. If False, drop solvent
            and isolated molecules but keep equal-sized interpenetrating nets.
        """
        # Get bond except metals.
        I, J, _ = covalent_neighbor_list(self.atoms)

        # Build MOF graph.
        graph = nx.Graph(zip(I, J))

        # Largest connected component is probably the MOF.
        ccs = sorted(nx.connected_components(graph), reverse=True, key=len)

        if len(ccs) < 2:
            # No interpenetration and no isolated molecules.
            indices = list(range(len(self.atoms)))
        elif remove_interpenetration:
            # If there are more than 1 connected components, use largest one.
            # But the second largest connected component may be not
            # interpenetration.
            indices = list(ccs[0])
        elif len(ccs[0]) == len(ccs[1]):
            indices = list(ccs[0] | ccs[1])
        else:
            indices = list(ccs[0])
        self.atoms = self.atoms[indices]

    @property
    def building_block_atom_indices(self):
        """Group the MOF's atoms into the fragments to be extracted.

        This is the core decomposition result: each set is one building
        block (a node or a linker), and ``make_building_block_atoms``
        turns each set into a capped fragment. Computed lazily on first
        access.

        Returns
        -------
        list[set]
            One set of atom indices per detected building block.
        """
        if not self.bb_found:
            self._find_building_block_atom_indices()
        return self._building_block_atom_indices

    @property
    def connecting_sites(self):
        """Return the atom indices that lie on a connecting bond.

        These are the atoms at the seams between fragments, i.e. the
        endpoints of the connecting bonds. ``make_building_block_atoms``
        uses them to decide where to graft the dummy ``X`` connection
        points onto an extracted fragment.

        Returns
        -------
        list[int]
            Sorted, de-duplicated atom indices appearing in any
            connecting bond.
        """
        return np.unique(self.connecting_bonds).tolist()

    @property
    def connecting_bonds(self):
        """Return the node-linker bonds that are cut to separate fragments.

        These are the seams of the MOF: severing them splits the structure
        into the individual building blocks, and their endpoints become the
        connection points (``X`` atoms) grafted onto each extracted
        fragment.

        Returns
        -------
        list[tuple[int, int]]
            Atom-index pairs, one per connecting bond.
        """

        if not self.bb_found:
            self._find_building_blocks()
        return self._connecting_bonds

    def extract_building_blocks(self):
        """Extract every building block and cache them for reuse.

        Materializes a capped fragment for each detected building block and
        stores the list on the ``building_blocks`` property.
        """
        n_bbs = len(self.building_block_atom_indices)
        self._building_blocks = [
            self.make_building_block_atoms(i) for i in range(n_bbs)
        ]

    @property
    def building_blocks(self):
        """Return the extracted building blocks, computing them on demand.

        This is the decomposer's payoff: the capped fragments that PORMAKE
        could feed back into the forward build pipeline as reusable
        building blocks.

        Returns
        -------
        list[ase.Atoms]
            One capped fragment per building block.
        """
        if not hasattr(self, '_building_blocks'):
            self.extract_building_blocks()
        return self._building_blocks

    def make_building_block_atoms(self, i):
        """Assemble the ``i``-th building block as a capped fragment.

        Gathers the atoms of the requested fragment, undoes any
        periodic-boundary cuts so the fragment is contiguous, and then
        caps each connecting site with a dummy ``X`` atom placed
        halfway along the severed bond. The result mirrors the
        connection-point convention of native PORMAKE building blocks,
        so the extracted fragment can be reused as one.

        Parameters
        ----------
        i : int
            Index into ``building_block_atom_indices`` selecting which
            fragment to build.

        Returns
        -------
        ase.Atoms
            The contiguous fragment with ``X_type`` connection-point
            atoms appended at each break point.

        Raises
        ------
        AssertionError
            If ``i`` is out of range for the discovered fragments.
        """
        assert len(self.building_block_atom_indices) > i

        indices = list(self.building_block_atom_indices[i])
        atoms = self.atoms[indices]

        # Remove pbc cuts.
        atoms = remove_pbc_cuts(atoms)

        connected_part_indices = np.where(
            [t in self.connecting_sites for t in indices]
        )[0]

        # Add X atom to connection site (now He)
        for ci in connected_part_indices:
            atom = copy.deepcopy(atoms[ci])
            connected_part_index = indices[ci]

            bonded_index = None
            bond = [
                t for t in self.connecting_bonds if connected_part_index in t
            ][0]

            if bond[0] == connected_part_index:
                bonded_index = bond[1]
            else:
                bonded_index = bond[0]

            vec = (
                self.atoms[bonded_index].position
                - self.atoms[connected_part_index].position
            )

            # Simple PBC consideration.
            norm_vec = np.matmul(vec, np.linalg.inv(atoms.cell))
            for i in range(3):
                if norm_vec[i] > 0.5:
                    norm_vec[i] -= 1.0
                if norm_vec[i] < -0.5:
                    norm_vec[i] += 1.0

            vec = np.matmul(atoms.cell, norm_vec) * 0.5
            atom.position = atom.position + vec
            atom.symbol = self.X_type
            atoms.append(atom)

        # hash_value = hash_atoms(atoms)
        # dimension = estimate_atoms_dimension(atoms)

        return atoms

    def _find_building_block_atom_indices(self):
        """Partition the MOF atoms into building-block fragments.

        Core of the decomposition. Builds the covalent-bond graph,
        removes the metal-coordination edges to expose candidate
        fragments, then identifies the bridge bonds linking metal
        clusters to organic linkers as the *connecting bonds* that mark
        where fragments meet. Self-linking linkers that reconnect to a
        single parent are merged back in when doing so does not change
        the fragment's periodic dimension. On completion this caches
        ``_building_block_atom_indices`` and ``_connecting_bonds`` and
        sets ``bb_found``.

        It is an experimental feature and may not be stable.
        """
        # Get full bond information.
        I, J, _ = covalent_neighbor_list(self.atoms)
        bond_list = [[] for _ in range(len(self.atoms))]

        # Build neighbor list as a list form.
        for i, j in zip(I, J):
            bond_list[i].append(j)

        # Get indices of metal atoms.
        metal_indices = [
            i for i, a in enumerate(self.atoms) if a.symbol in METAL_LIKE
        ]

        # Mark liking atom indices.
        liking_atom_indices = []
        for i in range(len(self.atoms)):
            if set(bond_list[i]) & set(metal_indices):
                liking_atom_indices.append(i)

        # Build MOF graph.
        graph = nx.Graph(zip(I, J))
        # Remove metal containing edges.
        metal_containing_edges = list(graph.edges(metal_indices))

        test_graph = graph.copy()
        test_graph.remove_edges_from(metal_containing_edges)
        result = []
        for cc in list(nx.connected_components(test_graph)):
            # Neglect single node components.
            if len(cc) == 1:
                continue

            # Construct graph of connected component.
            cc_graph = nx.subgraph(graph, cc).copy()

            # Get all bridges.
            bridges = list(nx.bridges(cc_graph))

            # Filter bridges.
            # Thie filter not filter out the self liking bridges.
            filtered_bridges = []
            for b in bridges:
                test_graph = cc_graph.copy()
                test_graph.remove_edge(*b)
                c1, c2 = list(nx.connected_components(test_graph))

                # Neglect no metal components.
                if not set(liking_atom_indices) & c1:
                    continue
                elif not set(liking_atom_indices) & c2:
                    continue

                # metal 연결된거 아니면 지운다 (continue)
                elif len(c1) == 1 and (c1 not in liking_atom_indices):
                    continue
                elif len(c2) == 1 and (c2 not in liking_atom_indices):
                    continue

                filtered_bridges.append(b)

            # Get first level bridges only.
            test_graph = cc_graph.copy()
            test_graph.remove_edges_from(filtered_bridges)
            test_ccs = list(nx.connected_components(test_graph))

            liking_ccs = []
            for test_cc in test_ccs:
                if set(liking_atom_indices) & test_cc:
                    liking_ccs.append(test_cc)

            first_level_bridges = set()
            for liking_cc in liking_ccs:
                for b in filtered_bridges:
                    if set(b) & liking_cc:
                        first_level_bridges.add(b)
            first_level_bridges = list(first_level_bridges)
            result += first_level_bridges
        first_level_bridges = result

        # Remove self liking ligands (like a ring).
        test_graph = graph.copy()
        # self.connecting_site_list = np.unique(first_level_bridges)
        test_graph.remove_edges_from(first_level_bridges)
        building_blocks = list(nx.connected_components(test_graph))
        # self._building_blocks = building_blocks

        # Merge self connecting linkers that form a path of bb to the same bb.
        index_to_bb = {}
        for i, bb in enumerate(building_blocks):
            for j in bb:
                index_to_bb[j] = i

        merging_dict = collections.defaultdict(list)
        for i, bb in enumerate(building_blocks):
            species = set(self.atoms[list(bb)].symbols)
            if species & set(METAL_LIKE):
                continue
            # Get connection point.
            connection_indices = []
            for j in bb:
                for k in graph.adj[j].keys():
                    if index_to_bb[k] == i:
                        continue
                    connection_indices.append(k)
            linked_bb_indices = [index_to_bb[_] for _ in connection_indices]
            if len(set(linked_bb_indices)) == 1:
                parent_bb_index = linked_bb_indices[0]
                merging_dict[parent_bb_index].append(i)

        children_indices = []
        new_bb = copy.deepcopy(building_blocks)
        for k, v in merging_dict.items():
            # Save index of child bb to remove later.
            tobemerged = set()
            for bb_index in v:
                tobemerged |= building_blocks[bb_index]
            new_bb[k] |= tobemerged
            # Check dimension changes.
            original_atoms = self.atoms[list(building_blocks[k])]
            original_dim = estimate_atoms_dimension(original_atoms)

            new_atoms = self.atoms[list(new_bb[k])]
            new_dim = estimate_atoms_dimension(new_atoms)

            if original_dim == new_dim:
                # Accept merging.
                children_indices += v
            else:
                # Reject merging.
                new_bb[k] = building_blocks[k]

        new_bb = [
            bb for i, bb in enumerate(new_bb) if i not in children_indices
        ]

        # Update building_block.
        building_blocks = new_bb

        # Remove improper first level bridges.
        index_to_bb = {}
        for i, bb in enumerate(building_blocks):
            for j in bb:
                index_to_bb[j] = i

        def is_valid(bridge):
            """Keep only bridges that still join two distinct fragments.

            After fragment merging some bridges become internal; reads the
            enclosing ``index_to_bb`` map to drop those.
            """
            i, j = bridge
            return index_to_bb[i] != index_to_bb[j]

        first_level_bridges = [b for b in first_level_bridges if is_valid(b)]

        self._connecting_bonds = first_level_bridges
        self._building_block_atom_indices = new_bb
        self.bb_found = True
