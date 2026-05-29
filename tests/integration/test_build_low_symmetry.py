from collections import Counter

import pytest

import pormake


@pytest.fixture(scope="module")
def low_symmetry_mof(database, builder):
    ith = database.get_topo("ith")
    node_bbs = {0: database.get_bb("N3"), 1: database.get_bb("N114")}
    edge_bbs = {(0, 1): database.get_bb("E41")}
    return builder.build_by_type(
        topology=ith, node_bbs=node_bbs, edge_bbs=edge_bbs
    )


def test_low_symmetry_atom_count(low_symmetry_mof):
    assert len(low_symmetry_mof.atoms) == 294


def test_low_symmetry_composition(low_symmetry_mof):
    composition = Counter(low_symmetry_mof.atoms.get_chemical_symbols())
    assert composition == Counter({"C": 216, "O": 48, "H": 24, "Ce": 6})
