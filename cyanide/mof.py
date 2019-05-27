import numpy as np

import ase
import ase.visualize
import ase.neighborlist

class MOF:
    def __init__(self, atoms, bonds):
        atoms = atoms.copy()
        atoms.wrap()
        self.atoms = atoms
        self.bonds = bonds.copy()

    def write(self, filename):
        """
        Write MOF in cif format.
        """
        with open(filename, "w") as f:
            f.write("TEST\n")

            f.write("_symmetry_space_group_name_H-M    P1\n")
            f.write("_symmetry_Int_Tables_number       1\n")
            f.write("_symmetry_cell_setting            triclinic\n")

            f.write("loop_\n")
            f.write("_symmetry_equiv_pos_as_xyz\n")
            f.write("'x, y, z'\n")

            a, b, c, alpha, beta, gamma = \
                self.atoms.get_cell_lengths_and_angles()

            f.write("_cell_length_a     {:.3f}\n".format(a))
            f.write("_cell_length_b     {:.3f}\n".format(b))
            f.write("_cell_length_c     {:.3f}\n".format(c))
            f.write("_cell_angle_alpha {:.3f}\n".format(alpha))
            f.write("_cell_angle_beta  {:.3f}\n".format(beta))
            f.write("_cell_angle_gamma {:.3f}\n".format(gamma))

            f.write("loop_\n")
            f.write("_atom_site_label\n")
            f.write("_atom_site_type_symbol\n")
            f.write("_atom_site_description\n")
            f.write("_atom_site_fract_x\n")
            f.write("_atom_site_fract_y\n")
            f.write("_atom_site_fract_z\n")
            f.write("_atom_type_partial_charge\n")

            symbols = self.atoms.symbols
            frac_coords = self.atoms.get_scaled_positions()
            for i, (sym, pos) in enumerate(zip(symbols, frac_coords)):
                label = "{}{}".format(sym, i)
                f.write("{} {} None {:.5f} {:.5f} {:.5f} 0.0\n".
                        format(label, sym, *pos))

            f.write("loop_\n")
            f.write("_geom_bond_atom_site_label_1\n")
            f.write("_geom_bond_atom_site_label_2\n")
            f.write("_geom_bond_distance\n")
            f.write("_geom_bond_site_symmetry_2\n")
            f.write("_ccdc_geom_bond_type\n") # ?????????

            # Get images
            I, J, S, D = ase.neighborlist.neighbor_list(
                            "ijSd", self.atoms, cutoff=6.0)
            image_dict = {}
            distance_dict = {}
            origin = np.array([5, 5, 5])
            for i, j, s, d in zip(I, J, S, D):
                image_dict[(i, j)] = s + origin
                distance_dict[(i, j)] = d

            for bond in self.bonds:
                i, j = bond

                sym = self.atoms.symbols[i]
                label_i = "{}{}".format(sym, i)

                sym = self.atoms.symbols[j]
                label_j = "{}{}".format(sym, j)

                distance = distance_dict[(i, j)]

                image = image_dict[(i, j)]

                if (image == origin).all():
                    f.write("{} {} {:.3f} . S\n".
                            format(label_i, label_j, distance))
                else:
                    f.write("{} {} {:.3f} 1_{}{}{} S\n".
                            format(label_i, label_j, distance, *image))

    def view(self):
        ase.visualize.view(self.atoms)
