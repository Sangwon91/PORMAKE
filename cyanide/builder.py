import numpy as np

from .scaler import Scaler
from .locator import Locator
from .mof import MOF

# bb: building block.
class Builder:
    def __init__(self):
        self.scaler = Scaler()
        self.locator = Locator()

    def build(self, topology, node_bbs, edge_bb=None, verbose=False):
        """
        The node_bbs must be given with proper order.
        Same as node type order in topology.
        """
        if verbose:
            echo = print
        else:
            echo = lambda x: None

        assert topology.n_node_types == len(node_bbs)

        # Calculate bonds before start.
        echo("Calculating bonds in building blocks...")
        for i in range(len(node_bbs)):
            node_bbs[i].bonds
        edge_bb.bonds

        echo("Calculating scaling factor...")
        # Get scaling factor.
        scaling_factor = self.scaler.calculate(topology, node_bbs, edge_bb)
        #print(scaling_factor)

        # Locate nodes and edges.
        located_bbs = [None for _ in topology.atoms]

        echo("Placing nodes...")
        # Locate nodes.
        for t, node_bb in enumerate(node_bbs):
            # t: node type.
            for i in topology.node_indices:
                if t != topology.get_node_type(i):
                    continue
                target = topology.local_structure(i)
                located_node, rms = self.locator.locate(target, node_bb)
                # Translate.
                center = topology.atoms[i].position * scaling_factor
                located_node.set_center(center)
                located_bbs[i] = located_node
                echo("Node {} is located, RMSD: {:.2E}".format(i, rms))

        # Locate edges.
        located_nodes = located_bbs[:topology.n_nodes]
        if edge_bb is not None:
            for i in topology.edge_indices:
                target = topology.local_structure(i)
                located_edge, rms = self.locator.locate(target, edge_bb)
                # Calculate edge center.
                # The center of edge can vary with different node sizes.
                # Assume this edge connects node 1 and node 2.
                n1, n2 = topology.neighbor_list[i]
                # Get normalized direction vector (1->2 direction).
                d = n2.distance_vector - n1.distance_vector
                d /= np.linalg.norm(d)
                # Get index of node 1 (not node 2).
                index = n1.index
                # Starting vector for edge position.
                o = located_nodes[index].center
                # Get half length of node 1.
                l1 = located_nodes[index].length
                # Get half length of edge.
                le = located_edge.length
                # Calculate center position. 1.5 for bond length.
                center = (l1+le+1.5)*d + o

                located_edge.set_center(center)
                located_bbs[i] = located_edge
                echo("Edge {} is located, RMSD: {:.2E}".format(i, rms))

        echo("Finding bonds in generated MOF...")
        echo("Finding bonds in building blocks...")
        # Build bonds of generated MOF.
        index_offsets = [None for _ in range(topology.n_all_points)]
        index_offsets[0] = 0
        for i, bb in enumerate(located_bbs[:-1]):
            if bb is None:
                continue
            index_offsets[i+1] = index_offsets[i] + bb.n_atoms

        bb_bonds = []
        for offset, bb in zip(index_offsets, located_bbs):
            if bb is None:
                continue
            bb_bonds.append(bb.bonds + offset)
        bb_bonds = np.concatenate(bb_bonds, axis=0)

        echo("Finding bonds between building blocks...")
        # Find bond between building blocks.
        # Calculate all permutations before.
        permutations = []
        for i, bb in enumerate(located_bbs):
            if bb is None:
                continue
            local_topo = topology.local_structure(i)
            local_bb = bb.local_structure()

            # local_topo.indices == local_bb.indices[perm]
            perm = local_topo.matching_permutation(local_bb)
            permutations.append(perm)

        bonds = []
        for j in topology.edge_indices:
            # i and j: edge index in topology
            n1, n2 = topology.neighbor_list[j]

            i1 = n1.index
            i2 = n2.index

            bb1 = located_bbs[i1]
            bb2 = located_bbs[i2]

            # Find bonded atom index for i1.
            for o, n in enumerate(topology.neighbor_list[i1]):
                # Check zero sum.
                s = n.distance_vector + n1.distance_vector
                s = np.linalg.norm(s)
                if s < 1e-3:
                    perm = permutations[i1]
                    a1 = bb1.connection_point_indices[perm][o]
                    a1 += index_offsets[i1]
                    break

            # Find bonded atom index for i2.
            for o, n in enumerate(topology.neighbor_list[i2]):
                # Check zero sum.
                s = n.distance_vector + n2.distance_vector
                s = np.linalg.norm(s)
                if s < 1e-3:
                    perm = permutations[i2]
                    a2 = bb2.connection_point_indices[perm][o]
                    a2 += index_offsets[i2]
                    break

            # Edge exists.
            if located_bbs[j] is not None:
                perm = permutations[j]
                e1, e2 = (
                    located_bbs[j].connection_point_indices[perm]
                    + index_offsets[j]
                )
                bonds.append((e1, a1))
                bonds.append((e2, a2))
            else:
                bonds.append((a1, a2))

            echo("Bonds on topology edge {} are connected.".format(j))

        bonds = np.array(bonds)

        # All bonds in generated MOF.
        all_bonds = np.concatenate([bb_bonds, bonds], axis=0)

        echo("Making MOF instance...")
        # Make full atoms from located building blocks.
        bb_atoms_list = [v.atoms for v in located_bbs if v is not None]

        mof_atoms = sum(bb_atoms_list[1:], bb_atoms_list[0])
        mof_atoms.set_pbc(True)
        mof_atoms.set_cell(topology.atoms.cell*scaling_factor)

        mof = MOF(mof_atoms, all_bonds)
        echo("Done.")

        return mof
