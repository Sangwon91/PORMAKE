"""Assemble building blocks onto a topology into a Framework.

This module hosts :class:`Builder`, the orchestrator that turns an
abstract net (a :class:`~pormake.topology.Topology`) plus a set of
molecular fragments (:class:`~pormake.building_block.BuildingBlock`)
into a concrete periodic structure (:class:`~pormake.framework.Framework`).

The build proceeds in stages: orient each node building block onto its
slot with the :class:`~pormake.locator.Locator` (checking against a
per-(node-type, bb) minimum RMSD and falling back to a chiral copy when
a fit is poor); record trivial edge permutations; rescale the topology
cell to the real building-block sizes with the
:class:`~pormake.scaler.Scaler`; relocate and translate the nodes onto
the scaled positions; lay edge linkers along the scaled edges; then find
all bonds, drop the "X" connection-point atoms while fusing the bonds
they implied, and wrap everything into a Framework.
"""

from collections import defaultdict

import numpy as np

from .framework import Framework
from .local_structure import LocalStructure
from .locator import Locator
from .log import logger
from .scaler import Scaler


# bb: building block.
class Builder:
    """Orchestrate assembly of a framework from blocks and a topology.

    Holds the two pluggable strategy objects the assembly depends on --
    a :class:`~pormake.locator.Locator` for orienting building blocks
    onto slots and a :class:`~pormake.scaler.Scaler` for resizing the
    topology cell -- and exposes :meth:`build` (and the per-type
    convenience wrappers) to run the full pipeline. Keeping locator and
    scaler injectable lets callers swap matching/scaling behavior without
    touching the orchestration logic.

    Parameters
    ----------
    locator : Locator, optional
        Strategy for finding the best orientation and permutation of a
        building block on a slot. A default :class:`Locator` is created
        when omitted.
    scaler : Scaler, optional
        Strategy for rescaling the topology unit cell to match real
        building-block sizes. A default :class:`Scaler` is created when
        omitted.

    Attributes
    ----------
    locator : Locator
        The orientation/permutation matcher used during the build.
    scaler : Scaler
        The cell-rescaling strategy used during the build.
    """

    def __init__(self, locator=None, scaler=None):
        """Store the locator and scaler, creating defaults if needed."""
        if locator is None:
            self.locator = Locator()
        else:
            self.locator = locator

        if scaler is None:
            self.scaler = Scaler()
        else:
            self.scaler = scaler

    def make_bbs_by_type(self, topology, node_bbs, edge_bbs=None):
        """Build the per-slot building-block list from type maps.

        Expands compact, type-keyed dictionaries into the dense
        slot-indexed list that :meth:`build` expects, copying each
        building block so every slot owns an independent instance.
        Specifying blocks by type (rather than per slot) is the
        ergonomic way to describe a structure, since all nodes of the
        same type and all edges of the same type usually share a block.
        Edge types absent from ``edge_bbs`` are logged and left empty
        (``None``), producing direct node-to-node bonds.

        Parameters
        ----------
        topology : Topology
            Topology whose slots are being populated; supplies the node
            and edge indices and their types.
        node_bbs : dict
            Maps a node type (int) to the building block to place on
            every node of that type.
        edge_bbs : dict, optional
            Maps an edge type (a tuple of two ints) to the linker
            building block for those edges. A value of ``None`` for a
            type leaves those edges empty. Defaults to no edge blocks.

        Returns
        -------
        list
            Length ``topology.n_slots`` list whose entry ``i`` is the
            (copied) building block for slot ``i``, or ``None`` for empty
            edge slots.
        """
        bbs = [None] * topology.n_slots

        for i in topology.node_indices:
            t = topology.node_types[i]
            bbs[i] = node_bbs[t].copy()

        if edge_bbs is None:
            # Empty dictionary.
            edge_bbs = {}

        # Log undefined edge building blocks.
        for t in topology.unique_edge_types:
            t = tuple(t)
            if t not in edge_bbs:
                logger.info(
                    "No edge building block for type %s in edge_bbs.", t
                )

        for i in topology.edge_indices:
            t = tuple(topology.edge_types[i])
            if t in edge_bbs:
                if edge_bbs[t] is None:
                    continue
                bbs[i] = edge_bbs[t].copy()

        return bbs

    def build_by_type(self, topology, node_bbs, edge_bbs=None, **kwargs):
        """Assemble a framework from type-keyed building-block maps.

        Convenience wrapper that turns the type-keyed ``node_bbs`` and
        ``edge_bbs`` dictionaries into a per-slot list via
        :meth:`make_bbs_by_type` and forwards it to :meth:`build`. This
        is the usual public entry point for users who think in terms of
        node and edge *types* rather than individual slots.

        Parameters
        ----------
        topology : Topology
            Net the framework is assembled on.
        node_bbs : dict
            Maps node type (int) to its building block.
        edge_bbs : dict, optional
            Maps edge type (tuple of two ints) to its linker building
            block. Defaults to no edge blocks.
        **kwargs
            Forwarded to :meth:`build` (e.g. ``accuracy``, ``wrap``,
            ``permutations``).

        Returns
        -------
        Framework
            The assembled framework.
        """
        bbs = self.make_bbs_by_type(topology, node_bbs, edge_bbs)
        return self.build(topology, bbs, **kwargs)

    def build(self, topology, bbs, permutations=None, **kwargs):
        """Assemble a framework from a per-slot list of building blocks.

        Runs the full assembly pipeline. For each node slot it finds the
        best orientation and connection-point permutation via the
        locator, comparing the achieved RMSD against a cached minimum for
        the (node type, building block) pair and retrying with higher
        accuracy and, if needed, a chiral copy of the block when the fit
        is more than 1% above that minimum. Edge slots are recorded with
        a trivial permutation. The topology cell is then scaled to the
        real block sizes, nodes are relocated and translated onto the
        scaled positions, and edge linkers are laid along the scaled
        edges. Finally all intra- and inter-block bonds are collected,
        the "X" connection-point atoms are removed while the bonds they
        implied are fused, and the result is returned as a Framework.

        Parameters
        ----------
        topology : Topology
            Net to assemble on. Node building blocks must correspond to
            the topology's node-type ordering.
        bbs : list
            Length ``topology.n_slots`` sequence where ``bbs[i]`` is the
            building block for node ``i`` (if ``i`` is a node index) or
            edge ``i`` (if an edge index). ``None`` leaves an edge empty.
        permutations : dict, optional
            Maps a slot index to a pre-determined connection-point
            permutation, bypassing the locator's search for that slot.
            Defaults to no pre-set permutations.
        **kwargs
            ``accuracy`` : int
                Euler-angle grid resolution for high-accuracy
                relocation (``max_n_slices``); default 6.
            ``wrap`` : bool
                Whether to wrap the framework into the unit cell;
                default ``True``.

        Returns
        -------
        Framework
            The assembled periodic framework, with bonds, bond types,
            and build metadata in ``info``.

        Raises
        ------
        Exception
            If a node placement yields an RMSD below 99% of the cached
            minimum (with that minimum above ``1e-3``), indicating the
            minimum RMSD was computed incorrectly.
        """

        # Parse keyword arguments.
        if "accuracy" in kwargs:
            max_n_slices = kwargs["accuracy"]
        else:
            max_n_slices = 6

        # Empty dictionary.
        if permutations is None:
            permutations = {}

        logger.debug("Builder.build starts.")

        # locator for bb locations.
        locator = self.locator

        # Locate nodes and edges.
        located_bbs = [None for _ in range(topology.n_slots)]

        _permutations = [None for _ in range(topology.n_slots)]
        for i, perm in permutations.items():
            _permutations[i] = np.array(perm)
        permutations = _permutations

        slot_min_rmsd = defaultdict(lambda: -1.0)
        # Locate nodes.
        for i in topology.node_indices:
            # Get bb.
            node_bb = bbs[i]
            # Get target.
            target = topology.local_structure(i)

            if permutations[i] is not None:
                perm = permutations[i]

                logger.info(
                    "Use given permutation for node slot index %d"
                    ", permutation: %s",
                    i,
                    perm,
                )

                located_node, rmsd = locator.locate_with_permutation(
                    target, node_bb, perm
                )

                logger.info(
                    "Pre-location of node slot %d, RMSD: %.2E",
                    i,
                    rmsd,
                )

                located_bbs[i] = located_node

                continue

            t = topology.get_node_type(i)
            # Calculate minimum RMSD of the slot.
            key = (t, node_bb.name)
            if slot_min_rmsd[key] < 0.0:
                rmsd = locator.calculate_rmsd(
                    target, node_bb, max_n_slices=max_n_slices
                )

                chiral_node_bb = node_bb.make_chiral_building_block()
                c_rmsd = locator.calculate_rmsd(target, chiral_node_bb)
                slot_min_rmsd[key] = min(rmsd, c_rmsd)
                logger.info(
                    "== Min RMSD of (node type: %s, node bb: %s): %.2E",
                    *key,
                    slot_min_rmsd[key],
                )
            # Only orientation.
            # Translations are applied after topology relexation.
            located_node, perm, rmsd = locator.locate(target, node_bb)
            logger.info(
                "Pre-location at node slot %d"
                ", (node type: %s, node bb: %s)"
                ", RMSD: %.2E",
                i,
                *key,
                rmsd,
            )
            # If RMSD is different from min RMSD relocate with high accuracy.
            # 1% error.
            ratio = rmsd / slot_min_rmsd[key]
            if ratio > 1.01:
                located_node, perm, rmsd = locator.locate(
                    target, node_bb, max_n_slices=max_n_slices
                )
                logger.info(
                    "RMSD > MIN_RMSD*1.01, relocate Node %d"
                    " with %d trial orientations, RMSD: %.2E",
                    i,
                    max_n_slices**3,
                    rmsd,
                )

            ratio = rmsd / slot_min_rmsd[key]
            if ratio > 1.01:
                # Make chiral building block.
                node_bb = node_bb.make_chiral_building_block()
                located_node, perm, rmsd = locator.locate(
                    target, node_bb, max_n_slices=max_n_slices
                )
                logger.info(
                    "RMSD > MIN_RMSD*1.01, relocate Node %d"
                    " with %d trial orientations and chiral building block"
                    ", RMSD: %.2E",
                    i,
                    max_n_slices**3,
                    rmsd,
                )

            # Critical error.
            if (ratio < 0.99) and (slot_min_rmsd[key] > 1e-3):
                message = (
                    "MIN_RMSD is not correct. "
                    "Topology: %s; "
                    "Slot: %s; "
                    "Building block: %s; "
                    "rmsd: %.3E." % (topology, key, node_bb, rmsd)
                )
                logger.error(message)
                raise Exception(message)

            located_bbs[i] = located_node
            # This information used in scaling of topology.
            permutations[i] = perm

        # Just append edges to the buidiling block slots.
        # There is no location in this stage.
        # This information is used in the scaling of topology.
        # All permutations are set to [0, 1] because the edges does not need
        # any permutation estimations for the locations.
        for e in topology.edge_indices:
            edge_bb = bbs[e]

            if edge_bb is None:
                continue

            if permutations[e] is not None:
                logger.info(
                    "Use given permutation for edge slot %d"
                    ", permutation: %s",
                    e,
                    permutations[e],
                )
                located_bbs[e] = edge_bb
                continue

            n1, n2 = topology.neighbor_list[e]

            i1 = n1.index
            i2 = n2.index

            if topology.node_types[i1] <= topology.node_types[i2]:
                perm = [0, 1]
            else:
                perm = [1, 0]

            located_bbs[e] = edge_bb
            permutations[e] = np.array(perm)

        # Scale topology.
        # Change topology to scaled topology.
        original_topology = topology
        topology, scaling_result = self.scaler.scale(
            topology=topology,
            bbs=located_bbs,
            perms=permutations,
            return_result=True,
        )

        rmsd_values = []
        # Relocate and translate node building blocks.
        for i in topology.node_indices:
            perm = permutations[i]
            node_bb = located_bbs[i]
            # Get target.
            target = topology.local_structure(i)
            # Orientation.
            located_node, rmsd = locator.locate_with_permutation(
                target, node_bb, perm
            )
            # Translation.
            centroid = topology.atoms.positions[i]
            located_node.set_centroid(centroid)

            # Update.
            located_bbs[i] = located_node

            t = topology.node_types[i]
            logger.info(
                "Location at node slot %d"
                ", (node type: %s, node bb: %s)"
                ", RMSD: %.2E",
                i,
                t,
                node_bb.name,
                rmsd,
            )

            rmsd_values.append(rmsd)
        rmsd_values = np.array(rmsd_values)

        # Thie helpers are so verbose. Anoying.
        def find_matched_atom_indices(e):
            """Return the two connection-point atoms an edge bonds to.

            For edge ``e``, locate within each adjacent node block the
            connection-point atom that faces the edge, so a linker can be
            bonded to it. Reads the enclosing-scope variables
            ``original_topology``, ``located_bbs``, and ``permutations``.

            Parameters
            ----------
            e : int
                Edge slot index in the (unscaled) topology.

            Returns
            -------
            a1, a2 : int
                Atom indices, within each neighboring node block, of the
                connection points matched to this edge's two ends.
            """
            topology = original_topology

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
                if s < 0.01:
                    perm = permutations[i1]
                    a1 = bb1.connection_point_indices[perm][o]
                    break

            # Find bonded atom index for i2.
            for o, n in enumerate(topology.neighbor_list[i2]):
                # Check zero sum.
                s = n.distance_vector + n2.distance_vector
                s = np.linalg.norm(s)
                if s < 0.01:
                    perm = permutations[i2]
                    a2 = bb2.connection_point_indices[perm][o]
                    break

            return a1, a2

        def calc_image(ni, nj, invc):
            """Compute the periodic image offset between two neighbors.

            Determine the integer unit-cell translation separating the
            two node ends of an edge, so an edge that wraps across a cell
            boundary is placed correctly. Reads the enclosing-scope
            ``topology``.

            Parameters
            ----------
            ni, nj : Neighbor
                The two neighbor records (node ends) of the edge.
            invc : numpy.ndarray
                Inverse of the topology cell matrix, used to convert the
                Cartesian offset into fractional (image) units.

            Returns
            -------
            numpy.ndarray
                The ``(3,)`` periodic image vector (integer-valued) of
                ``nj`` relative to ``ni``.
            """
            # Calculate image.
            # d = d_{ij}
            i = ni.index
            j = nj.index

            d = nj.distance_vector - ni.distance_vector

            ri = topology.atoms.positions[i]
            rj = topology.atoms.positions[j]

            image = (d - (rj - ri)) @ invc

            return image

        # Locate edges.
        logger.info("Start placing edges.")
        c = topology.atoms.cell
        invc = np.linalg.inv(topology.atoms.cell)
        for e in topology.edge_indices:
            edge_bb = located_bbs[e]
            # Neglect no edge cases.
            if edge_bb is None:
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
            d = r2 - r1 + image @ c

            # This may outside of the unit cell. Should be changed.
            centroid = r1 + 0.5 * d
            perm = permutations[e]

            target = LocalStructure(np.array([r1, r1 + d]), [i1, i2])
            located_edge, rmsd = locator.locate_with_permutation(
                target, edge_bb, perm
            )

            located_edge.set_centroid(centroid)
            located_bbs[e] = located_edge

            logger.debug(f"Edge {e}, RMSD: {rmsd:.2E}")

        logger.info("Start finding bonds in generated framework.")
        logger.info("Start finding bonds in building blocks.")
        # Build bonds of generated framework.
        index_offsets = [None for _ in range(topology.n_slots)]
        index_offsets[0] = 0
        for i, bb in enumerate(located_bbs[:-1]):
            if bb is None:
                index_offsets[i + 1] = index_offsets[i] + 0
            else:
                index_offsets[i + 1] = index_offsets[i] + bb.n_atoms

        bb_bonds = []
        bb_bond_types = []
        for offset, bb in zip(index_offsets, located_bbs):
            if bb is None:
                continue
            bb_bonds.append(bb.bonds + offset)
            bb_bond_types += bb.bond_types
        bb_bonds = np.concatenate(bb_bonds, axis=0)

        logger.info("Start finding bonds between building blocks.")

        # Find bond between building blocks.
        bonds = []
        bond_types = []
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
                bond_types += ["S", "S"]
                logger.debug(
                    "Bonds on topology edge %s are connected %s, %s.",
                    j,
                    bonds[-2],
                    bonds[-1],
                )
            else:
                bonds.append((a1, a2))
                bond_types += ["S"]
                logger.debug(
                    "Bonds on topology edge %s are connected %s.",
                    j,
                    bonds[-1],
                )

        bonds = np.array(bonds)

        # All bonds in generated framework.
        all_bonds = np.concatenate([bb_bonds, bonds], axis=0)
        all_bond_types = bb_bond_types + bond_types

        logger.info("Start making Framework instance.")
        # Make full atoms from located building blocks.
        bb_atoms_list = [v.atoms for v in located_bbs if v is not None]

        logger.debug("Merge list of atoms.")
        framework_atoms = sum(bb_atoms_list[1:], bb_atoms_list[0])
        logger.debug("Set cell and boundary.")
        framework_atoms.set_pbc(True)
        framework_atoms.set_cell(topology.atoms.cell)

        # Remove connection points (X) from the framework.
        count = 0
        new_indices = {}
        for a in framework_atoms:
            if a.symbol == "X":
                continue
            new_indices[a.index] = count
            count += 1

        def is_X(i):
            """Return whether atom ``i`` is an "X" connection point.

            Reads the enclosing-scope ``framework_atoms``; used to
            classify bonds while fusing and removing the dummy
            connection-point atoms.
            """
            return framework_atoms[i].symbol == "X"

        XX_bonds = []
        new_bonds = []
        new_bond_types = []
        X_neighbor_list = defaultdict(list)
        for (i, j), t in zip(all_bonds, all_bond_types):
            if is_X(i) and is_X(j):
                XX_bonds.append((i, j))
            elif is_X(i):
                X_neighbor_list[i] = j
            elif is_X(j):
                X_neighbor_list[j] = i
            else:
                new_bonds.append((i, j))
                new_bond_types.append(t)

        for i, j in XX_bonds:
            new_bonds.append((X_neighbor_list[i], X_neighbor_list[j]))
            new_bond_types.append("S")

        all_bonds = [(new_indices[i], new_indices[j]) for i, j in new_bonds]
        all_bonds = np.array(all_bonds)
        all_bond_types = new_bond_types

        del framework_atoms[[a.symbol == "X" for a in framework_atoms]]

        info = {
            "topology": topology,
            "located_bbs": located_bbs,
            "permutations": permutations,
            "relax_obj": scaling_result.fun,
            "max_rmsd": np.max(rmsd_values),
            "mean_rmsd": np.mean(rmsd_values),
        }

        if "wrap" not in kwargs:
            wrap = True
        else:
            wrap = kwargs["wrap"]

        framework = Framework(
            framework_atoms, all_bonds, all_bond_types, info=info, wrap=wrap
        )
        logger.info("Construction of framework done.")

        return framework
