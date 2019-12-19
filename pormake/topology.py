import os
import copy
import traceback
from pathlib import Path

import numpy as np

import ase
import ase.visualize

from .log import logger
from .utils import read_cgd
from .local_structure import LocalStructure
from .neighbor_list import Neighbor, NeighborList


class Topology:
    def __init__(self, cgd_file):
        self.atoms = read_cgd(filename=cgd_file)
        self.update_properties()

    def update_properties(self):
        """
        Calculate topology properties from information in self.atoms.
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
            self.neighbor_list = \
                NeighborList(self.atoms, method="nearest")

        if not self.check_validity():
            logger.error(
                "Topology parsing fails: {}".format(self.name)
            )
            raise Exception("Invalid cgd file: {}".format(self.name))
        else:
            logger.debug(
                "All coordination numbers are proper: {}".format(self.name)
            )

        # Calculate properties.
        self.calculate_properties()

    def check_coordination_numbers(self):
        for cn, ns in zip(self.atoms.info["cn"], self.neighbor_list):
            if cn != len(ns):
                return False
        return True

    def check_edge_zerosum(self):
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
        if not self.check_coordination_numbers():
            return False
        if not self.check_edge_zerosum():
            return False
        return True

    def copy(self):
        return copy.deepcopy(self)

    def calculate_properties(self):
        # Build indices of nodes and edges.
        self._node_indices = \
            np.argwhere(self.atoms.get_tags() != -1).reshape(-1)
        self._edge_indices = \
            np.argwhere(self.atoms.get_tags() == -1).reshape(-1)

        # Build node type.
        self._node_types = self.atoms.get_tags()

        # Build edge type.
        self._edge_types = [(-1, -1) for _ in range(self.n_all_points)]
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
        self._n_node_types = np.unique(self.node_types).shape[0] - 1
        self._n_edge_types = np.unique(self.edge_types, axis=0).shape[0] - 1

        self.cn = np.array([len(n) for n in self.neighbor_list])
        self.cn = self.cn[self.node_indices]
        self.unique_cn = []
        types = np.unique(self.node_types[self.node_indices])
        for t in types:
            i = np.argmax(self.node_types == t)
            self.unique_cn.append(self.cn[i])
        self.unique_cn = np.array(self.unique_cn)

    def local_structure(self, i):
        indices = []
        positions = []
        for n in self.neighbor_list[i]:
            indices.append(n.index)
            positions.append(n.distance_vector)

        return LocalStructure(positions, indices)

    def get_node_type(self, i):
        return self._node_types[i]

    def get_edge_type(self, i):
        return self._edge_types[i]

    @property
    def unique_local_structures(self):
        types = np.unique(self.node_types[self.node_indices])
        local_structures = []
        for t in types:
            i = np.argmax(self.node_types == t)
            local_structures.append(self.local_structure(i))
        return local_structures

    @property
    def unique_node_types(self):
        return np.unique(self.node_types[self.node_indices])

    @property
    def unique_edge_types(self):
        return np.unique(self.edge_types[self.edge_indices], axis=0)

    @property
    def node_types(self):
        return self._node_types

    @property
    def edge_types(self):
        return self._edge_types

    @property
    def n_all_points(self):
        return len(self.atoms)

    @property
    def node_indices(self):
        return self._node_indices

    @property
    def edge_indices(self):
        return self._edge_indices

    @property
    def n_nodes(self):
        return len(self.node_indices)

    @property
    def n_edges(self):
        return len(self.edge_indices)

    @property
    def n_node_types(self):
        return self._n_node_types

    @property
    def n_edge_types(self):
        return self._n_edge_types

    def get_neighbor_indices(self, i):
        return [n.index for n in self.neighbor_list[i]]

    def get_edge_length(self, i):
        n1, n2 = self.neighbor_list[i]
        diff = n1.distance_vector - n2.distance_vector
        return np.linalg.norm(diff)

    def view(self, show_edge_centers=True, repeat=1, **kwargs):
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
        atoms.set_cell(atoms.cell*scale)
        atoms.set_positions(s @ atoms.cell)

        # Visualize.
        r = repeat
        if isinstance(r, int):
            r = (r, r, r)
        ase.visualize.view(atoms, repeat=r, **kwargs)

    def __repr__(self):
        msg = "Topology {}, (".format(self.name)
        for cn in self.unique_cn:
            msg += f"{cn},"
        msg = msg[:-1] + ")-cn, num edge types: {}".format(self.n_edge_types)

        return msg

    def write_cif(self, filename, with_edge_atoms=False, *args, **kwargs):
        """
        Write topology in cif format.
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
            logger.error(
                "CIF writing fails with error: %s",
                traceback.format_exc(),
            )
            # Remove invalid cif file.
            logger.error("Remove invalid CIF: %s", path)
            os.remove(str(path))

    def _write_cif_with_edge_atoms(self, path, scale=1.0):
        stem = path.stem.replace(" ", "_")

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

            a, b, c, alpha, beta, gamma = \
                self.atoms.get_cell_lengths_and_angles()

            f.write("_cell_length_a     {:.3f}\n".format(a*scale))
            f.write("_cell_length_b     {:.3f}\n".format(b*scale))
            f.write("_cell_length_c     {:.3f}\n".format(c*scale))
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
                ## 57: atomic number of the first lanthanide element.
                # 7: Nitrogen.
                return ase.Atom(tag+7).symbol

            def tags2symbols(tags):
                return [tag2symbol(tag) for tag in tags]

            tags = self.atoms.get_tags()
            symbols = tags2symbols(tags)
            frac_coords = self.atoms.get_scaled_positions()
            for i, (sym, pos) in enumerate(zip(symbols, frac_coords)):
                label = "{}{}".format(sym, i)
                f.write("{} {} {:.5f} {:.5f} {:.5f} 0.0\n".
                        format(label, sym, *pos))

            f.write("loop_\n")
            f.write("_geom_bond_atom_site_label_1\n")
            f.write("_geom_bond_atom_site_label_2\n")
            f.write("_geom_bond_distance\n")
            f.write("_geom_bond_site_symmetry_2\n")
            f.write("_ccdc_geom_bond_type\n") # ?????????

            origin = np.array([5, 5, 5])

            eps = 1e-3
            invcell = np.linalg.inv(self.atoms.get_cell())
            bond_info = []
            for i in self.node_indices:
                for edge in self.neighbor_list[i]:
                    j = edge.index
                    d = edge.distance_vector

                    rj = self.atoms[j].position
                    ri = self.atoms[i].position
                    rij = rj - ri

                    image = np.dot(d-rij, invcell)
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
                    f.write("{} {} {:.3f} . {}\n".
                        format(label_i, label_j, distance, bond_type)
                    )
                else:
                    f.write("{} {} {:.3f} 1_{}{}{} {}\n".
                        format(label_i, label_j, distance, *image, bond_type)
                    )

    def _write_cif_without_edge_atoms(self, path, scale=1.0):
        stem = path.stem.replace(" ", "_")

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

            a, b, c, alpha, beta, gamma = \
                self.atoms.get_cell_lengths_and_angles()

            f.write("_cell_length_a     {:.3f}\n".format(a*scale))
            f.write("_cell_length_b     {:.3f}\n".format(b*scale))
            f.write("_cell_length_c     {:.3f}\n".format(c*scale))
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
                ## 57: atomic number of the first lanthanide element.
                # 7: Nitrogen.
                return ase.Atom(tag+7).symbol

            def tags2symbols(tags):
                return [tag2symbol(tag) for tag in tags]

            node_indices = self.node_indices
            tags = self.atoms.get_tags()[node_indices]
            symbols = tags2symbols(tags)
            frac_coords = self.atoms.get_scaled_positions()[node_indices]
            for i, (sym, pos) in enumerate(zip(symbols, frac_coords)):
                label = "{}{}".format(sym, i)
                f.write("{} {} {:.5f} {:.5f} {:.5f} 0.0\n".
                        format(label, sym, *pos))

            f.write("loop_\n")
            f.write("_geom_bond_atom_site_label_1\n")
            f.write("_geom_bond_atom_site_label_2\n")
            f.write("_geom_bond_distance\n")
            f.write("_geom_bond_site_symmetry_2\n")
            f.write("_ccdc_geom_bond_type\n") # ?????????

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

                    image = np.dot(2*d-rij, invcell)
                    image = np.around(image).astype(np.int32)

                    distance = np.linalg.norm(2*d)

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
                    f.write("{} {} {:.3f} . {}\n".
                        format(label_i, label_j, distance, bond_type)
                    )
                else:
                    f.write("{} {} {:.3f} 1_{}{}{} {}\n".
                        format(label_i, label_j, distance, *image, bond_type)
                    )

    def __mul__(self, m):
        atoms = self.atoms.copy()

        old_cn = atoms.info["cn"]
        old_tags = atoms.get_tags()

        n = len(atoms)

        atoms = atoms * m

        new_cn = [old_cn[i] for i in atoms.get_tags()]
        new_tags = [old_tags[i] for i in atoms.get_tags()]

        atoms.info["cn"] = new_cn
        atoms.set_tags(new_tags)

        new_topology = self.copy()
        new_topology.atoms = atoms
        new_topology.update_properties()

        return new_topology
