from collections import defaultdict

import numpy as np

from .log import logger
from .mof import MOF
from .scaler import Scaler
from .locator import Locator
from .local_structure import LocalStructure

# bb: building block.
class Builder:
    def __init__(self):
        self.scaler = Scaler()
        self.locator = Locator()

    def build(self, topology, node_bbs, edge_bbs=None, custom_edge_bbs=None):
        """
        The node_bbs must be given with proper order.
        Same as node type order in topology.

        Inputs:
            custom_edge_bbs: Custom edge building blocks at specific edge
                index e. It is a dict, keys are edge index and values are
                building block.
        """
        logger.debug("Builder.build starts.")

        if edge_bbs is None:
            edge_bbs = defaultdict(lambda: None)
        else:
            edge_bbs = defaultdict(lambda: None, edge_bbs)

        # make empty dictionary.
        if custom_edge_bbs is None:
            custom_edge_bbs = {}

        assert topology.n_node_types == len(node_bbs)

        # Calculate bonds before start.
        logger.info("Start pre-calculation of bonds in building blocks.")
        for node in node_bbs:
            node.bonds

        for edge in edge_bbs.values():
            if edge is None:
                continue
            edge.bonds

        logger.info("Start topology scaling.")
        # Get scaled topology.
        scaled_topology = \
            self.scaler.scale(topology, node_bbs, edge_bbs, custom_edge_bbs)

        # Replace topology to scaled_topology
        topology = scaled_topology

        # Locate nodes and edges.
        located_bbs = [None for _ in range(topology.n_all_points)]

        logger.info("Start placing nodes.")
        # Locate nodes.
        for t, node_bb in enumerate(node_bbs):
            # t: node type.
            for i in topology.node_indices:
                if t != topology.get_node_type(i):
                    continue
                target = topology.local_structure(i)
                located_node, rms = self.locator.locate(target, node_bb)
                # Translate.
                center = topology.atoms[i].position
                located_node.set_center(center)
                located_bbs[i] = located_node
                logger.info(f"Node {i} is located, RMSD: {rms:.2E}")

        # Calculate matching permutations of nodes.
        # Permutation of edges are matched later.
        logger.debug("Start finding maching permuation of nodes.")
        permutations = [None for _ in range(topology.n_all_points)]
        for i in topology.node_indices:
            bb = located_bbs[i]
            local_topo = topology.local_structure(i)
            local_bb = bb.local_structure()

            # pos of local_topo.indices ~ pos of local_bb.indices[perm]
            perm = local_topo.matching_permutation(local_bb)
            permutations[i] = perm

        def find_matched_atom_indices(e):
            """
            Inputs:
                e: Edge index.

            External variables:
                topology, located_bbs, permutations.
            """
            # i and j: edge index in topology
            n1, n2 = topology.neighbor_list[e]

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
                    break

            # Find bonded atom index for i2.
            for o, n in enumerate(topology.neighbor_list[i2]):
                # Check zero sum.
                s = n.distance_vector + n2.distance_vector
                s = np.linalg.norm(s)
                if s < 1e-3:
                    perm = permutations[i2]
                    a2 = bb2.connection_point_indices[perm][o]
                    break

            return a1, a2

        def calc_image(ni, nj, invc):
            """
            Calculate image number.
            External variables:
                topology.
            """
            # Calculate image.
            # d = d_{ij}
            i = ni.index
            j = nj.index

            d = nj.distance_vector - ni.distance_vector

            ri = topology.atoms.positions[i]
            rj = topology.atoms.positions[j]

            image = (d - (rj-ri)) @ invc

            return image

        # Locate edges.
        logger.info("Start placing edges.")
        c = topology.atoms.cell
        invc = np.linalg.inv(topology.atoms.cell)
        for t, edge_bb in edge_bbs.items():
            if edge_bb is None:
                continue
            for e in topology.edge_indices:
                if e in custom_edge_bbs:
                    continue

                ti, tj = topology.get_edge_type(e)
                if t != (ti, tj):
                    continue

                n1, n2 = topology.neighbor_list[e]

                i1 = n1.index
                i2 = n2.index

                bb1 = located_bbs[i1]
                bb2 = located_bbs[i2]

                a1, a2 = find_matched_atom_indices(e)

                r1 = bb1.atoms.positions[a1]
                r2 = bb2.atoms.positions[a2]

                image = calc_image(n1, n2, invc)
                d = r2 - r1 + image@c

                center = r1 + 0.5*d

                target = LocalStructure(np.array([r1, r1+d]), [i1, i2])
                located_edge, rms = self.locator.locate(target, edge_bb)

                located_edge.set_center(center)
                located_bbs[e] = located_edge

                logger.info(f"Edge {e} is located, RMSD: {rms:.2E}")

        # Locate custom edges.
        logger.info("Start placing custom edges.")
        for e, edge_bb in custom_edge_bbs.items():
            n1, n2 = topology.neighbor_list[e]

            i1 = n1.index
            i2 = n2.index

            bb1 = located_bbs[i1]
            bb2 = located_bbs[i2]

            a1, a2 = find_matched_atom_indices(e)

            r1 = bb1.atoms.positions[a1]
            r2 = bb2.atoms.positions[a2]

            image = calc_image(n1, n2, invc)
            d = r2 - r1 + image@c

            center = r1 + 0.5*d

            target = LocalStructure(np.array([r1, r1+d]), [i1, i2])
            located_edge, rms = self.locator.locate(target, edge_bb)

            located_edge.set_center(center)
            located_bbs[e] = located_edge

            logger.info(f"Custom edge {e} is located, RMSD: {rms:.2E}")

        # Calculate edge matching permutations
        logger.debug("Start finding maching permuation of edges.")
        for i in topology.edge_indices:
            bb = located_bbs[i]
            if bb is None:
                continue
            local_topo = topology.local_structure(i)
            local_bb = bb.local_structure()

            # pos of local_topo.indices ~ pos of local_bb.indices[perm]
            perm = local_topo.matching_permutation(local_bb)
            permutations[i] = perm

        logger.info("Start finding bonds in generated MOF.")
        logger.info("Start finding bonds in building blocks.")
        # Build bonds of generated MOF.
        index_offsets = [None for _ in range(topology.n_all_points)]
        index_offsets[0] = 0
        for i, bb in enumerate(located_bbs[:-1]):
            if bb is None:
                index_offsets[i+1] = index_offsets[i] + 0
            else:
                index_offsets[i+1] = index_offsets[i] + bb.n_atoms

        bb_bonds = []
        for offset, bb in zip(index_offsets, located_bbs):
            if bb is None:
                continue
            bb_bonds.append(bb.bonds + offset)
        bb_bonds = np.concatenate(bb_bonds, axis=0)

        logger.info("Start finding bonds between building blocks.")

        # Find bond between building blocks.
        bonds = []
        for j in topology.edge_indices:
            a1, a2 = find_matched_atom_indices(j)

            # i and j: edge index in topology
            n1, n2 = topology.neighbor_list[j]
            i1 = n1.index
            i2 = n2.index
            a1 += index_offsets[i1]
            a2 += index_offsets[i2]

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

            logger.info(f"Bonds on topology edge {j} are connected.")

        bonds = np.array(bonds)

        # All bonds in generated MOF.
        all_bonds = np.concatenate([bb_bonds, bonds], axis=0)

        logger.info("Start Making MOF instance.")
        # Make full atoms from located building blocks.
        bb_atoms_list = [v.atoms for v in located_bbs if v is not None]

        logger.debug("Merge list of atoms.")
        mof_atoms = sum(bb_atoms_list[1:], bb_atoms_list[0])
        logger.debug("Set cell and boundary.")
        mof_atoms.set_pbc(True)
        mof_atoms.set_cell(topology.atoms.cell)

        mof = MOF(mof_atoms, all_bonds, wrap=True)
        logger.info("Construction of MOF done.")

        return mof
