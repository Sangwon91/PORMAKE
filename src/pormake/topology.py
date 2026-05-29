"""The abstract net that organizes a framework.

This module defines :class:`Topology`, the blueprint that says how many
building-block slots exist and how they connect, without any chemistry
attached yet. Read from a ``.cgd`` file, it classifies slots into node and
edge types, exposes each slot's local structure for the
:class:`~pormake.locator.Locator`, and is rescaled by the
:class:`~pormake.scaler.Scaler` before the :class:`~pormake.builder.Builder`
populates it with real building blocks.
"""
import copy
import functools
import os
import sys
import traceback
from pathlib import Path

import ase
import ase.visualize
import numpy as np

from .local_structure import LocalStructure
from .log import logger
from .neighbor_list import NeighborList
from .utils import read_cgd


class Topology:
    """Represent an abstract periodic net that scaffolds a framework.

    A ``Topology`` is the blueprint stage of the PORMAKE pipeline: it is
    read from a ``.cgd`` file describing a reticular net and holds the
    positions, unit cell, and connectivity of that net. Geometry lives in
    an ``ase.Atoms`` object where carbon ("C") atoms are node (vertex)
    slots and oxygen ("O") atoms are edge-center slots; integer tags on
    each atom encode its type (tag ``>= 0`` for a node type, tag ``< 0``
    for an edge). On construction the net's connectivity is built into a
    ``NeighborList`` and validated (coordination numbers and edge
    zero-sum), then per-slot and per-type properties are cached.

    Downstream stages consume this object: the ``Scaler`` rescales its
    cell to fit chosen building blocks, the ``Locator`` aligns each
    ``BuildingBlock`` to the ``LocalStructure`` of the slot it occupies,
    and the ``Builder`` assembles the result into a ``Framework``. A
    "slot" is any node or edge position a building block can fill; an
    "edge type" is the sorted pair of node-types an edge connects.

    Parameters
    ----------
    cgd_file : str or pathlib.Path
        Path to the ``.cgd`` file describing the net to parse.
    local_structure_func : callable, optional
        Normalization function forwarded to each ``LocalStructure``
        produced for a slot; controls how raw neighbor directions are
        normalized before the ``Locator`` matches a building block. If
        ``None``, the ``LocalStructure`` default is used.

    Attributes
    ----------
    atoms : ase.Atoms
        Net geometry: "C" atoms are node slots, "O" atoms are
        edge-centers, integer tags encode node/edge type, and
        ``atoms.info`` carries ``"name"``, ``"spacegroup"`` and ``"cn"``.
    name : str
        Net identifier read from the ``.cgd`` file.
    spacegroup : str
        Space-group label of the net.
    neighbor_list : NeighborList
        Node-to-edge-center connectivity used for validation and for
        building local structures.
    local_structure_func : callable or None
        Normalization function applied when constructing each slot's
        ``LocalStructure``.
    cn : numpy.ndarray
        Per-slot coordination numbers (from ``atoms.info["cn"]``).
    unique_cn : numpy.ndarray
        One coordination number per unique node type, aligned with
        ``unique_node_types``.
    """

    def __init__(self, cgd_file, local_structure_func=None):
        """Parse ``cgd_file`` and build/validate the net's properties."""
        self.atoms = read_cgd(filename=cgd_file)
        self.update_properties()
        self.local_structure_func = local_structure_func

    def update_properties(self):
        """Parse, validate, and cache all net properties from ``self.atoms``.

        Reads the net name and space group, builds a ``NeighborList`` with
        the fast distance method and, if validation fails, retries with
        the more tolerant nearest-two method. Once a valid connectivity is
        found, ``calculate_properties`` is called to derive node/edge
        indices, types, and counts. This is the single entry point that
        refreshes everything after the geometry changes (e.g. after
        ``__mul__`` replaces ``self.atoms``).

        Raises
        ------
        Exception
            If neither connectivity method yields a valid net (i.e. the
            ``.cgd`` file is malformed).
        """

        # Save additional information.
        self.name = self.atoms.info["name"]
        self.spacegroup = self.atoms.info["spacegroup"]

        self.neighbor_list = NeighborList(self.atoms, method="distance")
        if not self.check_validity():
            logger.debug(
                "{}: Distance based parsing fails.. "
                "Try nearest two method.".format(self.name)
            )
            self.neighbor_list = NeighborList(self.atoms, method="nearest")

        if not self.check_validity():
            logger.error("Topology parsing fails: {}".format(self.name))
            raise Exception("Invalid cgd file: {}".format(self.name))
        else:
            logger.debug(
                "All coordination numbers are proper: {}".format(self.name)
            )

        # Calculate properties.
        self.calculate_properties()

    def check_coordination_numbers(self):
        """Verify each slot's neighbor count matches its expected CN.

        Compares the coordination numbers declared in the ``.cgd`` file
        (``atoms.info["cn"]``) against the number of neighbors the
        ``NeighborList`` actually found for each slot. A mismatch means
        the connectivity was parsed incorrectly.

        Returns
        -------
        bool
            ``True`` if every slot's neighbor count equals its declared
            coordination number, ``False`` otherwise.
        """
        for cn, ns in zip(self.atoms.info["cn"], self.neighbor_list):
            if cn != len(ns):
                return False
        return True

    def check_edge_zerosum(self):
        """Verify each edge-center sits midway between its two nodes.

        A correctly parsed edge-center has exactly two node neighbors
        whose displacement vectors are equal and opposite, so they sum to
        zero (within a small tolerance). A non-zero sum signals that the
        edge-center is not centered on its bond and the connectivity is
        wrong.

        Returns
        -------
        bool
            ``True`` if every edge-center's two neighbor displacements
            cancel to within ``1e-3``, ``False`` otherwise.
        """
        eps = 1e-3
        edge_indices = np.argwhere(self.atoms.get_tags() == -1).reshape(-1)
        for e in edge_indices:
            n1, n2 = self.neighbor_list[e]

            zerosum = n1.distance_vector + n2.distance_vector
            zerosum = np.abs(zerosum)

            if (zerosum > eps).any():
                return False
        return True

    def check_validity(self):
        """Check that the parsed connectivity is self-consistent.

        Combines the two structural checks: coordination numbers must
        match the declared values and every edge-center must be balanced
        between its two nodes. ``update_properties`` uses the result to
        decide whether the current connectivity method succeeded.

        Returns
        -------
        bool
            ``True`` only if both ``check_coordination_numbers`` and
            ``check_edge_zerosum`` pass.
        """
        if not self.check_coordination_numbers():
            return False
        if not self.check_edge_zerosum():
            return False
        return True

    def copy(self):
        """Return a deep, independent copy of this topology.

        Returns
        -------
        Topology
            A fully detached duplicate (geometry, connectivity, and cached
            properties) safe to mutate without affecting the original;
            used, for example, by ``__mul__`` to build a supercell.
        """
        return copy.deepcopy(self)

    def calculate_properties(self):
        """Derive cached node/edge indices, types, and counts.

        From the validated ``ase.Atoms`` tags and ``NeighborList`` this
        precomputes the data the rest of the pipeline queries repeatedly:
        which slots are nodes vs. edges, each slot's node type, each
        edge's (sorted) pair of adjacent node-types, the number of unique
        node and edge types, and the coordination number of each unique
        node type. Called by ``update_properties`` after a valid
        connectivity is established.
        """
        # Build indices of nodes and edges.
        self._node_indices = np.argwhere(self.atoms.get_tags() >= 0).reshape(-1)
        self._edge_indices = np.argwhere(self.atoms.get_tags() < 0).reshape(-1)

        # Build node type.
        self._node_types = self.atoms.get_tags()

        # Build edge type.
        self._edge_types = [(-1, -1) for _ in range(self.n_slots)]
        for i in self.edge_indices:
            n = self.neighbor_list[i]

            i0 = n[0].index
            i1 = n[1].index

            t0 = self.get_node_type(i0)
            t1 = self.get_node_type(i1)

            # Sort.
            if t0 > t1:
                t0, t1 = t1, t0

            self._edge_types[i] = (t0, t1)
        self._edge_types = np.array(self._edge_types)

        # Calculate the number of node and edge types.
        self._n_node_types = np.unique(
            self.node_types[self.node_indices]
        ).shape[0]
        self._n_edge_types = np.unique(
            self.edge_types[self.edge_indices], axis=0
        ).shape[0]

        # self.cn = np.array([len(n) for n in self.neighbor_list])
        # self.cn = self.cn[self.node_indices]
        self.cn = self.atoms.info["cn"]
        self.unique_cn = []
        types = np.unique(self.node_types[self.node_indices])
        for t in types:
            i = np.argmax(self.node_types == t)
            self.unique_cn.append(self.cn[i])
        self.unique_cn = np.array(self.unique_cn)

    def local_structure(self, i):
        """Return the normalized neighbor directions around slot ``i``.

        Gathers the displacement vectors from slot ``i`` to each of its
        neighbors and packages them as a ``LocalStructure`` -- the
        normalized set of directions that defines the slot's connection
        geometry. This is the geometric "currency" the ``Locator``
        matches against a building block's own ``LocalStructure`` (via
        RMSD minimization) to orient and place that block into slot ``i``.

        Parameters
        ----------
        i : int
            Slot index (node or edge-center) whose local geometry to
            extract.

        Returns
        -------
        LocalStructure
            Neighbor directions of slot ``i``, normalized with
            ``self.local_structure_func``, tagged with the neighbor slot
            indices.
        """
        indices = []
        positions = []
        for n in self.neighbor_list[i]:
            indices.append(n.index)
            positions.append(n.distance_vector)

        return LocalStructure(
            positions, indices, normalization_func=self.local_structure_func
        )

    def get_node_type(self, i):
        """Return the node type of slot ``i``.

        Parameters
        ----------
        i : int
            Slot index.

        Returns
        -------
        int
            The slot's integer tag/node type (``>= 0`` for nodes; for an
            edge-center this is its negative edge tag).
        """
        return self._node_types[i]

    def get_edge_type(self, i):
        """Return the (sorted) node-type pair an edge slot connects.

        Parameters
        ----------
        i : int
            Slot index. Meaningful for edge slots; node slots carry the
            placeholder ``(-1, -1)``.

        Returns
        -------
        numpy.ndarray
            The two adjacent node types as a sorted ``(t0, t1)`` pair,
            which is how edge types are identified and grouped.
        """
        return self._edge_types[i]

    @property
    def unique_local_structures(self):
        """Return one representative ``LocalStructure`` per node type.

        Picks the first slot of each unique node type and returns its
        local structure, giving the distinct connection geometries the
        net contains. The ``Locator`` only needs to solve the alignment
        problem once per node type, so this list (aligned with
        ``unique_node_types``) is the natural unit of work.

        Returns
        -------
        list of LocalStructure
            One local structure per unique node type, in ascending
            node-type order.
        """
        types = np.unique(self.node_types[self.node_indices])
        local_structures = []
        for t in types:
            i = np.argmax(self.node_types == t)
            local_structures.append(self.local_structure(i))
        return local_structures

    @property
    def unique_node_types(self):
        """Return the distinct node types present in the net.

        Returns
        -------
        numpy.ndarray
            Sorted unique node-type tags; the set of vertex kinds that
            each need a building block assigned.
        """
        return np.unique(self.node_types[self.node_indices])

    @property
    def unique_edge_types(self):
        """Return the distinct edge types present in the net.

        Returns
        -------
        numpy.ndarray
            Unique ``(t0, t1)`` node-type pairs over all edge slots; the
            set of edge kinds that may each take a (linker) building
            block.
        """
        return np.unique(self.edge_types[self.edge_indices], axis=0)

    @property
    def node_types(self):
        """Return the per-slot node-type array.

        Returns
        -------
        numpy.ndarray
            The integer tag of every slot (length ``n_slots``); for edge
            slots this is the negative edge tag. Index with
            ``node_indices`` to restrict to true nodes.
        """
        return self._node_types

    @property
    def edge_types(self):
        """Return the per-slot edge-type array.

        Returns
        -------
        numpy.ndarray
            For each slot, the sorted ``(t0, t1)`` pair of adjacent node
            types; node slots hold the placeholder ``(-1, -1)``. Index
            with ``edge_indices`` to restrict to true edges.
        """
        return self._edge_types

    @property
    def n_all_points(self):
        """Return the total number of slots in the net.

        Returns
        -------
        int
            Count of all atoms in ``self.atoms``, i.e. nodes plus
            edge-centers.
        """
        return len(self.atoms)

    @property
    def n_slots(self):
        """Return the number of fillable slots (alias of ``n_all_points``).

        Returns
        -------
        int
            Total node and edge-center slots a building block can occupy.
        """
        return self.n_all_points

    @property
    def node_indices(self):
        """Return the slot indices that are nodes (vertices).

        Returns
        -------
        numpy.ndarray
            Indices into ``self.atoms`` of slots with tag ``>= 0``.
        """
        return self._node_indices

    @property
    def edge_indices(self):
        """Return the slot indices that are edge-centers.

        Returns
        -------
        numpy.ndarray
            Indices into ``self.atoms`` of slots with tag ``< 0``.
        """
        return self._edge_indices

    @property
    def n_nodes(self):
        """Return the number of node (vertex) slots.

        Returns
        -------
        int
            Count of node slots in the net.
        """
        return len(self.node_indices)

    @property
    def n_edges(self):
        """Return the number of edge-center slots.

        Returns
        -------
        int
            Count of edge slots in the net.
        """
        return len(self.edge_indices)

    @property
    def n_node_types(self):
        """Return the number of distinct node types.

        Returns
        -------
        int
            How many unique vertex kinds the net contains; the number of
            independent building-block choices for nodes.
        """
        return self._n_node_types

    @property
    def n_edge_types(self):
        """Return the number of distinct edge types.

        Returns
        -------
        int
            How many unique edge kinds (node-type pairs) the net
            contains.
        """
        return self._n_edge_types

    def get_neighbor_indices(self, i):
        """Return the slot indices adjacent to slot ``i``.

        Parameters
        ----------
        i : int
            Slot index whose neighbors to list.

        Returns
        -------
        list of int
            Indices of the slots bonded to slot ``i``, drawn from the
            ``NeighborList``.
        """
        return [n.index for n in self.neighbor_list[i]]

    def get_edge_length(self, i):
        """Return the through-space length of edge slot ``i``.

        Computes the distance between the edge-center's two node
        neighbors from their displacement vectors. ``Scaler`` uses such
        lengths when sizing the cell to fit the chosen building blocks.

        Parameters
        ----------
        i : int
            Edge slot index (an edge-center with exactly two neighbors).

        Returns
        -------
        float
            Euclidean distance between the edge's two adjacent nodes.
        """
        n1, n2 = self.neighbor_list[i]
        diff = n1.distance_vector - n2.distance_vector
        return np.linalg.norm(diff)

    def view(self, show_edge_centers=True, *args, **kwargs):
        """Open an interactive ASE viewer of the net.

        Renders the abstract net for visual inspection. The unit cell is
        expanded so vertices and bonds are easy to see; when edge-centers
        are shown their "O" atoms are relabeled "F" purely so the viewer
        draws the node-edge bonds clearly. The geometry of ``self`` is not
        modified (a copy is displayed).

        Parameters
        ----------
        show_edge_centers : bool, optional
            If ``True`` (default), display edge-center slots (relabeled
            for bond visibility); if ``False``, remove them and show only
            nodes.
        *args, **kwargs
            Forwarded to ``ase.visualize.view``.
        """
        atoms = self.atoms.copy()

        if show_edge_centers:
            scale = 3
            # Replace symbol O to F for bond visualization.
            symbols = np.array(atoms.get_chemical_symbols())
            symbols[symbols == "O"] = "F"
            atoms.set_chemical_symbols(symbols)
        else:
            scale = 2
            del atoms[atoms.symbols == "O"]

        # Expand cell for the visualization.
        s = atoms.get_scaled_positions()
        atoms.set_cell(atoms.cell * scale)
        atoms.set_positions(s @ atoms.cell)

        ase.visualize.view(atoms, *args, **kwargs)

    def __repr__(self):
        """Return a one-line summary naming the net, its CNs, and edge types.

        Returns
        -------
        str
            Text of the form ``Topology <name>, (cn, ...)-cn, num edge
            types: <n>`` summarizing the coordination numbers of the
            unique node types and the number of edge types.
        """
        msg = "Topology {}, (".format(self.name)
        for cn in self.unique_cn:
            msg += f"{cn},"
        msg = msg[:-1] + ")-cn, num edge types: {}".format(self.n_edge_types)

        return msg

    def write_cif(self, filename, with_edge_atoms=False, *args, **kwargs):
        """Write the abstract net to a CIF file for visualization.

        Serializes the net (in P1 symmetry) so it can be inspected in
        crystallography viewers. Each slot type is mapped to a distinct
        chemical symbol and node-edge connectivity is emitted as CIF
        bonds. The ``.cif`` suffix is added if missing; if writing fails,
        the partially written file is removed and the error is logged.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output path; a ``.cif`` suffix is appended when absent.
        with_edge_atoms : bool, optional
            If ``True``, write both node and edge-center atoms; if
            ``False`` (default), write only nodes and connect them
            directly through the edge midpoints.
        *args, **kwargs
            Forwarded to the underlying ``_write_cif_*`` helper (e.g.
            ``scale``).
        """
        path = Path(filename).resolve()
        if not path.parent.exists():
            logger.error(f"Path {path} does not exist.")

        # Add suffix if not exists.
        if path.suffix != ".cif":
            path = path.with_suffix(".cif")

        try:
            if with_edge_atoms:
                self._write_cif_with_edge_atoms(path, *args, **kwargs)
            else:
                self._write_cif_without_edge_atoms(path, *args, **kwargs)
        except Exception as e:
            logger.exception(e)
            logger.error(
                "CIF writing fails with error: %s",
                traceback.format_exc(),
            )
            # Remove invalid cif file.
            logger.error("Remove invalid CIF: %s", path)
            os.remove(str(path))

    def _write_cif_with_edge_atoms(self, path, scale=1.0):
        """Write a CIF containing both node and edge-center atoms.

        Emits every slot (nodes and edge-centers) as an atom and records
        each node-to-edge-center bond, with periodic image labels for
        bonds that cross the cell boundary. Used by ``write_cif`` when
        ``with_edge_atoms=True``.

        Parameters
        ----------
        path : pathlib.Path
            Resolved output path (already suffixed ``.cif``).
        scale : float, optional
            Multiplier applied to cell lengths and bond distances
            (default ``1.0``).
        """
        # stem = path.stem.replace(" ", "_")

        with path.open("w") as f:
            # Write information comments.
            f.write("# Topology {}\n".format(self.name))

            # Real CIF information starts.
            f.write("data_{}\n".format(self.name))

            f.write("_symmetry_space_group_name_H-M    P1\n")
            f.write("_symmetry_Int_Tables_number       1\n")
            f.write("_symmetry_cell_setting            triclinic\n")

            f.write("loop_\n")
            f.write("_symmetry_equiv_pos_as_xyz\n")
            f.write("'x, y, z'\n")

            (
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
            ) = self.atoms.get_cell_lengths_and_angles()

            f.write("_cell_length_a     {:.3f}\n".format(a * scale))
            f.write("_cell_length_b     {:.3f}\n".format(b * scale))
            f.write("_cell_length_c     {:.3f}\n".format(c * scale))
            f.write("_cell_angle_alpha  {:.3f}\n".format(alpha))
            f.write("_cell_angle_beta   {:.3f}\n".format(beta))
            f.write("_cell_angle_gamma  {:.3f}\n".format(gamma))

            f.write("loop_\n")
            f.write("_atom_site_label\n")
            f.write("_atom_site_type_symbol\n")
            f.write("_atom_site_fract_x\n")
            f.write("_atom_site_fract_y\n")
            f.write("_atom_site_fract_z\n")
            f.write("_atom_type_partial_charge\n")

            def tag2symbol(tag):
                """Map a slot tag to a distinct chemical symbol."""
                # 57: atomic number of the first lanthanide element.
                # 7: Nitrogen.
                return ase.Atom(tag + 7).symbol

            def tags2symbols(tags):
                """Map a sequence of slot tags to chemical symbols."""
                return [tag2symbol(tag) for tag in tags]

            tags = self.atoms.get_tags()
            symbols = tags2symbols(tags)
            frac_coords = self.atoms.get_scaled_positions()
            for i, (sym, pos) in enumerate(zip(symbols, frac_coords)):
                label = "{}{}".format(sym, i)
                f.write(
                    "{} {} {:.5f} {:.5f} {:.5f} 0.0\n".format(label, sym, *pos)
                )

            f.write("loop_\n")
            f.write("_geom_bond_atom_site_label_1\n")
            f.write("_geom_bond_atom_site_label_2\n")
            f.write("_geom_bond_distance\n")
            f.write("_geom_bond_site_symmetry_2\n")
            f.write("_ccdc_geom_bond_type\n")  # ?????????

            origin = np.array([5, 5, 5])

            # eps = 1e-3
            invcell = np.linalg.inv(self.atoms.get_cell())
            bond_info = []
            for i in self.node_indices:
                for edge in self.neighbor_list[i]:
                    j = edge.index
                    d = edge.distance_vector

                    rj = self.atoms[j].position
                    ri = self.atoms[i].position
                    rij = rj - ri

                    image = np.dot(d - rij, invcell)
                    image = np.around(image).astype(np.int32)

                    distance = np.linalg.norm(d)

                    bond_info.append((i, j, image, distance))

            for i, j, image, distance in bond_info:
                sym = symbols[i]
                label_i = "{}{}".format(sym, i)

                sym = symbols[j]
                label_j = "{}{}".format(sym, j)

                bond_type = "S"

                image = origin + image
                distance *= scale

                if (image == origin).all():
                    f.write(
                        "{} {} {:.3f} . {}\n".format(
                            label_i, label_j, distance, bond_type
                        )
                    )
                else:
                    f.write(
                        "{} {} {:.3f} 1_{}{}{} {}\n".format(
                            label_i, label_j, distance, *image, bond_type
                        )
                    )

    def _write_cif_without_edge_atoms(self, path, scale=1.0):
        """Write a CIF containing only node atoms.

        Emits just the node (vertex) slots and connects each pair of
        nodes that share an edge-center directly, doubling the
        node-to-edge displacement to span the full node-node bond and
        labeling periodic images as needed. Used by ``write_cif`` when
        ``with_edge_atoms=False``.

        Parameters
        ----------
        path : pathlib.Path
            Resolved output path (already suffixed ``.cif``).
        scale : float, optional
            Multiplier applied to cell lengths and bond distances
            (default ``1.0``).

        Raises
        ------
        Exception
            If an edge-center's reciprocal node neighbor cannot be matched
            while reconstructing node-node bonds.
        """
        # stem = path.stem.replace(" ", "_")

        with path.open("w") as f:
            # Write information comments.
            f.write("# Topology {}\n".format(self.name))

            # Real CIF information starts.
            f.write("data_{}\n".format(self.name))

            f.write("_symmetry_space_group_name_H-M    P1\n")
            f.write("_symmetry_Int_Tables_number       1\n")
            f.write("_symmetry_cell_setting            triclinic\n")

            f.write("loop_\n")
            f.write("_symmetry_equiv_pos_as_xyz\n")
            f.write("'x, y, z'\n")

            (
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
            ) = self.atoms.get_cell_lengths_and_angles()

            f.write("_cell_length_a     {:.3f}\n".format(a * scale))
            f.write("_cell_length_b     {:.3f}\n".format(b * scale))
            f.write("_cell_length_c     {:.3f}\n".format(c * scale))
            f.write("_cell_angle_alpha  {:.3f}\n".format(alpha))
            f.write("_cell_angle_beta   {:.3f}\n".format(beta))
            f.write("_cell_angle_gamma  {:.3f}\n".format(gamma))

            f.write("loop_\n")
            f.write("_atom_site_label\n")
            f.write("_atom_site_type_symbol\n")
            f.write("_atom_site_fract_x\n")
            f.write("_atom_site_fract_y\n")
            f.write("_atom_site_fract_z\n")
            f.write("_atom_type_partial_charge\n")

            def tag2symbol(tag):
                """Map a slot tag to a distinct chemical symbol."""
                # 57: atomic number of the first lanthanide element.
                # 7: Nitrogen.
                return ase.Atom(tag + 7).symbol

            def tags2symbols(tags):
                """Map a sequence of slot tags to chemical symbols."""
                return [tag2symbol(tag) for tag in tags]

            node_indices = self.node_indices
            tags = self.atoms.get_tags()[node_indices]
            symbols = tags2symbols(tags)
            frac_coords = self.atoms.get_scaled_positions()[node_indices]
            for i, (sym, pos) in enumerate(zip(symbols, frac_coords)):
                label = "{}{}".format(sym, i)
                f.write(
                    "{} {} {:.5f} {:.5f} {:.5f} 0.0\n".format(label, sym, *pos)
                )

            f.write("loop_\n")
            f.write("_geom_bond_atom_site_label_1\n")
            f.write("_geom_bond_atom_site_label_2\n")
            f.write("_geom_bond_distance\n")
            f.write("_geom_bond_site_symmetry_2\n")
            f.write("_ccdc_geom_bond_type\n")  # ?????????

            origin = np.array([5, 5, 5])

            eps = 1e-3
            invcell = np.linalg.inv(self.atoms.get_cell())
            bond_info = []
            for i in self.node_indices:
                for edge in self.neighbor_list[i]:
                    k = edge.index
                    d = edge.distance_vector
                    found = False
                    for node in self.neighbor_list[k]:
                        abs_diff = np.abs(d - node.distance_vector)
                        if (abs_diff < eps).all():
                            found = True
                            break

                    if not found:
                        raise Exception("..?")

                    j = node.index

                    if i > j:
                        continue

                    rj = self.atoms[j].position
                    ri = self.atoms[i].position
                    rij = rj - ri

                    image = np.dot(2 * d - rij, invcell)
                    image = np.around(image).astype(np.int32)

                    distance = np.linalg.norm(2 * d)

                    bond_info.append((i, j, image, distance))

            for i, j, image, distance in bond_info:
                sym = symbols[i]
                label_i = "{}{}".format(sym, i)

                sym = symbols[j]
                label_j = "{}{}".format(sym, j)

                bond_type = "S"

                image = origin + image
                distance *= scale

                if (image == origin).all():
                    f.write(
                        "{} {} {:.3f} . {}\n".format(
                            label_i, label_j, distance, bond_type
                        )
                    )
                else:
                    f.write(
                        "{} {} {:.3f} 1_{}{}{} {}\n".format(
                            label_i, label_j, distance, *image, bond_type
                        )
                    )

    def __mul__(self, m):
        """Return a supercell of this net replicated along the cell axes.

        Multiplies the underlying ``ase.Atoms`` to tile the net, carries
        each slot's coordination number over to the replicated slots, and
        rebuilds all derived properties on the new (larger) topology.
        Useful for generating frameworks bigger than the primitive net.
        The original topology is left unchanged.

        Parameters
        ----------
        m : int or sequence of int
            Repetition along the cell vectors, in any form ``ase.Atoms``
            multiplication accepts (e.g. ``2`` or ``(2, 2, 1)``).

        Returns
        -------
        Topology
            A new topology representing the requested supercell.
        """
        atoms = self.atoms.copy()

        old_cn = atoms.info["cn"]
        old_tags = atoms.get_tags()

        tag2cn = {tag: cn for tag, cn in zip(old_tags, old_cn)}

        # n = len(atoms)

        atoms = atoms * m

        new_cn = [tag2cn[i] for i in atoms.get_tags()]
        atoms.info["cn"] = new_cn

        new_topology = self.copy()
        new_topology.atoms = atoms
        new_topology.update_properties()

        return new_topology

    def __rmul__(self, m):
        """Return a supercell, supporting ``m * topology`` ordering.

        Parameters
        ----------
        m : int or sequence of int
            Repetition along the cell vectors; see ``__mul__``.

        Returns
        -------
        Topology
            The same supercell ``__mul__`` would produce.
        """
        return self * m

    def describe(
        self, symmetry_edge_type=False, slot_info=False, file=sys.stdout
    ):
        """Print a human-readable summary of the net's structure.

        Writes a formatted report covering the net name and space group,
        slot/type counts, the slot indices grouped by node type and by
        edge type, and optionally a symmetry-based edge grouping and a
        per-slot listing. Intended as a quick, readable inventory of what
        the net offers when choosing building-block assignments.

        Parameters
        ----------
        symmetry_edge_type : bool, optional
            If ``True``, also print edges grouped by their symmetry-based
            (single negative-tag) type. Default ``False``.
        slot_info : bool, optional
            If ``True``, also print a line per slot with its type and
            adjacent slots. Default ``False``.
        file : file-like, optional
            Destination stream for the report (default ``sys.stdout``).
        """
        def comma_numbers(nums):
            """Join numbers into a comma-separated string."""
            return ", ".join([str(n) for n in nums])

        def split_list(l, size):
            """Chunk a list into sublists of at most ``size`` items."""
            out = []
            inner = []
            for e in l:
                inner.append(e)
                if len(inner) == size:
                    out.append(inner)
                    inner = []
            if inner:
                out.append(inner)
            return out

        # Alias for print function.
        p = functools.partial(print, file=file)

        p("=" * 79)
        p("Topology %s" % self.name)
        p("Spacegroup: %s" % self.spacegroup)
        p("-" * 79)

        p(
            "# of slots: %d (%d nodes, %d edges)"
            % (self.n_slots, self.n_nodes, self.n_edges)
        )
        p("# of node types: %d" % self.n_node_types)
        p("# of edge types: %d" % self.n_edge_types)
        p("")

        p("-" * 79)
        p("Node type information")
        p("-" * 79)
        for t, cn in zip(self.unique_node_types, self.unique_cn):
            p("Node type: %d, CN: %d" % (t, cn))
            indices = np.argwhere(self.node_types == t).reshape(-1).tolist()
            split_indices = split_list(indices, 10)
            p("  slot indices: %s" % comma_numbers(split_indices[0]))
            for indices in split_indices[1:]:
                p("                %s" % comma_numbers(indices))
        p("")

        p("-" * 79)
        p("Edge type information (adjacent node types) ")
        p("-" * 79)
        for t in self.unique_edge_types:
            p("Edge type: (%d, %d)" % (*t,))
            condition = self.edge_types == t
            condition = np.all(condition, axis=1)
            indices = np.argwhere(condition).reshape(-1).tolist()
            split_indices = split_list(indices, 10)
            p("  slot indices: %s" % comma_numbers(split_indices[0]))
            for indices in split_indices[1:]:
                p("                %s" % comma_numbers(indices))

        if symmetry_edge_type:
            p("")
            p("-" * 79)
            p("Edge type information (symmetry) ")
            p("-" * 79)
            for t in np.unique(-self.node_types[self.edge_indices]):
                t = -t
                p("Edge type: %d" % t)
                condition = self.node_types == t
                indices = np.argwhere(condition).reshape(-1).tolist()
                split_indices = split_list(indices, 10)
                p("  slot indices: %s" % comma_numbers(split_indices[0]))
                for indices in split_indices[1:]:
                    p("                %s" % comma_numbers(indices))

        if slot_info:
            p("")
            p("-" * 79)
            p("Slot information")
            p("-" * 79)
            for i in range(self.n_slots):
                p("[Slot %d]" % i, end=" ")
                adjacent_slot_indices = comma_numbers(
                    [n.index for n in self.neighbor_list[i]]
                )
                if i in self.node_indices:
                    cn = self.cn[i]
                    p(
                        "node type: %d, CN: %d, adjecent slot: %s"
                        % (self.node_types[i], cn, adjacent_slot_indices)
                    )
                if i in self.edge_indices:
                    p(
                        "edge type: (%d, %d) or %d, adjecent slot: %s"
                        % (
                            *self.edge_types[i],
                            self.node_types[i],
                            adjacent_slot_indices,
                        )
                    )

        p("=" * 79)
