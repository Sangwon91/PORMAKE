from pathlib import Path

import numpy as np

import ase
import ase.visualize
import ase.neighborlist

class MOF:
    def __init__(self, atoms, bonds, wrap=True):
        atoms = atoms.copy()
        #atoms.wrap(eps=1e-4)
        if wrap:
            # Wrap atoms.
            eps = 1e-4
            scaled_positions = atoms.get_scaled_positions()
            scaled_positions = \
                np.where(scaled_positions > 1.0-eps,
                         np.zeros_like(scaled_positions),
                         scaled_positions,
                )
            atoms.set_scaled_positions(scaled_positions)

        # Save data to attributes.
        self.atoms = atoms
        self.bonds = bonds.copy()

    def write_cif(self, filename):
        """
        Write MOF in cif format.
        """

        path = Path(filename)
        if path.suffix != ".cif":
            path = path.with_suffix(".cif")

        stem = path.stem.replace(" ", "_")

        with path.open("w") as f:
            f.write("data_{}\n".format(stem))

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

            symbols = self.atoms.symbols
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

            # Get images and distances.
            I, J, S, D = ase.neighborlist.neighbor_list(
                            "ijSd", self.atoms, cutoff=8.0)
            image_dict = {}
            distance_dict = {}
            origin = np.array([5, 5, 5])
            for i, j, s, d in zip(I, J, S, D):
                image_dict[(i, j)] = origin + s
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
