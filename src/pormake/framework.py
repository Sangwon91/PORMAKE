"""Represent and serialize the final assembled porous framework.

This module defines :class:`Framework`, the terminal product of the
PORMAKE pipeline. After :class:`~pormake.builder.Builder` has oriented,
scaled, and fused all building blocks, it hands the merged periodic
structure here. The framework couples an :class:`ase.Atoms` object (cell
plus atomic positions and partial charges) with an explicit bond list
and per-bond chemical types, and remembers the build provenance in its
``info`` dictionary. Its main job downstream is to emit a P1 CIF file --
including periodic-image symmetry codes on bonds -- that other
crystallography and simulation tools can read.
"""

import copy
import os
import traceback
from collections import defaultdict
from pathlib import Path

import ase
import ase.neighborlist
import ase.visualize
import numpy as np

from .log import logger


class Framework:
    """Hold a fully assembled periodic framework and write it to CIF.

    Bundles the geometry, bonding, and build metadata produced at the
    end of an assembly run into one object so the result can be
    inspected, visualized, and serialized. Bonds are tracked explicitly
    (rather than re-derived from distances) because the builder knows
    exactly which atoms were fused when the "X" connection points were
    removed, which a distance-based guess could get wrong.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms of the assembled framework, carrying the unit cell,
        periodic boundary conditions, and per-atom partial charges.
    bonds : numpy.ndarray
        Integer array of shape ``(n_bonds, 2)`` listing bonded atom
        index pairs.
    bond_types : list of str
        Chemical bond type label for each entry in ``bonds`` (e.g.
        ``"S"`` for single).
    info : dict
        Build provenance: the (scaled) ``topology``, the list of
        ``located_bbs``, the chosen ``permutations``, the scaler's
        ``relax_obj`` value, and the ``max_rmsd``/``mean_rmsd`` of the
        node placements. Used when writing CIF comments.
    wrap : bool, optional
        If ``True`` (default), fold atoms into the unit cell and nudge
        them off the cell boundary via :meth:`wrap` on construction.

    Attributes
    ----------
    atoms : ase.Atoms
        A copy of the input atoms (possibly wrapped).
    bonds : numpy.ndarray
        A copy of the input bond pairs.
    bond_types : list of str
        A deep copy of the input bond type labels.
    info : dict
        A deep copy of the input build metadata.
    """

    def __init__(self, atoms, bonds, bond_types, info, wrap=True):
        """Store copies of the geometry, bonds, and metadata."""
        # Save data to attributes.
        self.atoms = atoms.copy()
        self.bonds = bonds.copy()
        self.bond_types = copy.deepcopy(bond_types)
        self.info = copy.deepcopy(info)

        if wrap:
            self.wrap()

    def wrap(self):
        """Fold atoms into the cell and off its boundary.

        Exists to keep every atom inside the unit cell and, critically,
        to avoid fractional coordinates that sit exactly on a cell
        boundary (0 or 1). Building blocks placed by the builder can land
        slightly outside the cell or precisely on a face; many CIF
        parsers and visualizers either reject or double-count atoms whose
        fractional coordinates equal 0 or 1. This routine repeatedly
        shifts coordinates back into the ``[0, 1)`` range and then nudges
        any value within ``1e-3`` of a boundary inward (to ``0.0001`` or
        ``0.9999``) so the emitted structure is numerically safe
        downstream. Modifies ``self.atoms`` in place.
        """
        # Cleanup errors.
        s = self.atoms.get_scaled_positions()
        while (s >= 1.0).any() and (s < 0.0).any():
            s = np.where(
                s < 0.0,
                s + np.ones_like(s),
                s,
            )
            s = np.where(
                s >= 1.0,
                s - np.ones_like(s),
                s,
            )

        s[np.abs(s - 1) < 1e-3] = 0.9999
        s[np.abs(s) < 1e-3] = 0.0001

        self.atoms.set_scaled_positions(s)

    def write_cif(self, filename):
        """Write the framework to a CIF file.

        Public, fault-tolerant entry point for serialization. It
        normalizes the path, appends a ``.cif`` suffix when missing, and
        delegates the actual formatting to :meth:`_write_cif`. If writing
        raises, the error is logged and the partially written, invalid
        file is removed so no corrupt CIF is left behind.

        Parameters
        ----------
        filename : str or pathlib.Path
            Destination path. A ``.cif`` suffix is added if absent.
        """
        path = Path(filename).resolve()
        if not path.parent.exists():
            logger.error(f"Path {path} does not exist.")

        # Add suffix if not exists.
        if path.suffix != ".cif":
            path = path.with_suffix(".cif")

        try:
            self._write_cif(path)
        except Exception as e:
            logger.error(e)
            logger.error(
                "CIF writing fails with error: %s",
                traceback.format_exc(),
            )
            # Remove invalid cif file.
            logger.error("Remove invalid CIF: %s", path)
            os.remove(str(path))

    def _write_cif(self, path):
        """Emit the framework as a P1 CIF at ``path``.

        Writes the full CIF document: provenance comments (topology,
        building blocks, relaxation objective, RMSD statistics), the P1
        symmetry block, cell lengths and angles, the atom-site loop with
        fractional coordinates and partial charges, and a bond loop. For
        bonds, a neighbor-list search recovers each pair's minimum-image
        distance and the periodic image of the second atom; bonds that
        cross a cell boundary are tagged with a ``1_pqr`` symmetry code
        encoding that image so the periodic connectivity is preserved.

        Parameters
        ----------
        path : pathlib.Path
            Destination file path (already resolved and suffixed by
            :meth:`write_cif`).
        """
        stem = path.stem.replace(" ", "_")

        with path.open("w") as f:
            # Write information comments.
            _topology = self.info["topology"]
            info_str = "# Generated by pormake.\n"
            info_str += "# {}\n".format(_topology)
            info_str += "# Building blocks:\n"
            for i in _topology.node_indices:
                t = _topology.node_types[i]
                bb = self.info["located_bbs"][i]
                info_str += "#     Node {}, Type {}, {}\n".format(i, t, bb)
            for i in _topology.edge_indices:
                t = tuple(_topology.edge_types[i])
                bb = self.info["located_bbs"][i]
                info_str += "#     Edge {}, Type {}, {}\n".format(i, t, bb)
            f.write(info_str)
            f.write(
                "# Relax obj value: {:.3f}\n".format(self.info["relax_obj"])
            )
            f.write("# Max RMSD: {:.3f}\n".format(self.info["max_rmsd"]))
            f.write("# Mean RMSD: {:.3f}\n".format(self.info["mean_rmsd"]))

            f.write("data_{}\n".format(stem))

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
            charges = self.atoms.get_initial_charges()
            frac_coords = self.atoms.get_scaled_positions()
            for i, (sym, pos, charge) in enumerate(zip(symbols, frac_coords, charges)):
                label = "{}{}".format(sym, i)
                f.write(
                    "{} {} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(label, sym, *pos, charge)
                )

            f.write("loop_\n")
            f.write("_geom_bond_atom_site_label_1\n")
            f.write("_geom_bond_atom_site_label_2\n")
            f.write("_geom_bond_distance\n")
            f.write("_geom_bond_site_symmetry_2\n")
            f.write("_ccdc_geom_bond_type\n")  # ?????????

            # Make bond type dict.
            bond_type_dict = {}
            for (i, j), t in zip(self.bonds, self.bond_types):
                bond_type_dict[(i, j)] = t

            # Get images and distances.
            I, J, S, D = ase.neighborlist.neighbor_list(
                "ijSd", self.atoms, cutoff=6.0
            )
            image_dict = {}
            distance_dict = defaultdict(lambda: 1e30)
            origin = np.array([5, 5, 5])
            for i, j, s, d in zip(I, J, S, D):
                # Take minimum image only.
                if d < distance_dict[(i, j)]:
                    image_dict[(i, j)] = origin + s
                    distance_dict[(i, j)] = d

            for bond in self.bonds:
                i, j = bond

                sym = self.atoms.symbols[i]
                label_i = "{}{}".format(sym, i)

                sym = self.atoms.symbols[j]
                label_j = "{}{}".format(sym, j)

                distance = distance_dict[(i, j)]
                bond_type = bond_type_dict[(i, j)]

                image = image_dict[(i, j)]

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

    def view(self, *args, **kwargs):
        """Open the framework atoms in an ASE viewer.

        Thin convenience wrapper around :func:`ase.visualize.view` for
        quick visual inspection of the assembled structure. Positional
        and keyword arguments are forwarded unchanged to ASE.
        """
        ase.visualize.view(self.atoms, *args, **kwargs)
