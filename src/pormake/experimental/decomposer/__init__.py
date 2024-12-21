"""Experimental MOF Decomposer module. It's refactored from the legacy code used
in the PORMAKE paper. It is an experimental feature and may not be stable.
"""
import collections
import copy
from pathlib import Path

import ase
import networkx as nx
import numpy as np

from ...utils import METAL_LIKE, covalent_neighbor_list


def hash_atoms(atoms: ase.Atoms, complexity: int = 6):
    """Hashes an ase.Atoms object into a unique integer. The hash is based on
    the adjacency matrix of the covalent bonds and the atomic numbers of the
    atoms. The hash is invariant to the order of the atoms in the atoms object.
    It is an experimental feature and may not be stable.

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
    """Estimates the dimension of the atoms object in periodic system. Is is
    used to estimate the dimension of MOFs or building blocks in the MOF. For
    current version of PORMAKE, only building blocks with 0 dimension are
    supported. It is an experimental feature and may not be stable.

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
    def __init__(self, cif, X_type='X'):
        """MOF decomposer class. It is an experimental feature and may not be
        stable.

        TODO:
            * Custom bond information (connectivity and bond types).

        Parameters
        ----------
        cif : str
            The path to the CIF file of the MOF.
        X_type : str
            The symbol of the X atom. It is used to identify the connection
            sites. Default is 'X'.
        """
        self.atoms = ase.io.read(cif)
        self.name = Path(cif).stem
        self.bb_found = False
        self.X_type = X_type

    def view(self, *args, **kwargs):
        ase.visualize.view(self.atoms, *args, **kwargs)

    def cleanup(self, remove_interpenetration=True):
        """Removes interpenetration and isolated molecules from the MOF.
        It is an experimental feature and may not be stable.

        Parameters
        ----------
        remove_interpenetration : bool
            If True, removes interpenetration. If False, removes only isolated
            molecules.
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
        """Returns the atom indices of the building blocks.

        Returns
        -------
        list[set]:
            The list of atom indices of the building blocks.
        """
        if not self.bb_found:
            self._find_building_block_atom_indices()
        return self._building_block_atom_indices

    @property
    def connecting_sites(self):
        return np.unique(self.connecting_bonds).tolist()

    @property
    def connecting_bonds(self):
        """Returns the indices of connecting bonds between metal nodes and
        organic linkes of the MOF.

        Returns
        -------
        list[tuple[int, int]]
            The list of connecting bonds.
        """

        if not self.bb_found:
            self._find_building_blocks()
        return self._connecting_bonds

    def extract_building_blocks(self):
        """Extracts building blocks from the MOF. Building blocks are stored in
        the self.building_blocks property.
        """
        n_bbs = len(self.building_block_atom_indices)
        self._building_blocks = [
            self.make_building_block_atoms(i) for i in range(n_bbs)
        ]

    @property
    def building_blocks(self):
        """Returns the building blocks of the MOF.

        Returns
        -------
        list[ase.Atoms]
            The list of building blocks in ase.Atoms format.
        """
        if not hasattr(self, '_building_blocks'):
            self.extract_building_blocks()
        return self._building_blocks

    def make_building_block_atoms(self, i):
        """Makes building block atoms from the MOF."""
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
        """Finds building blocks and connecting bonds of the MOF. It is an
        experimental feature and may not be stable."""
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
            i, j = bridge
            return index_to_bb[i] != index_to_bb[j]

        first_level_bridges = [b for b in first_level_bridges if is_valid(b)]

        self._connecting_bonds = first_level_bridges
        self._building_block_atom_indices = new_bb
        self.bb_found = True
