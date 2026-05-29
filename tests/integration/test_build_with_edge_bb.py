from collections import Counter

import pytest

import pormake


@pytest.fixture(scope="module")
def hkust1_with_edge(database, builder):
    tbo = database.get_topo("tbo")
    node_bbs = {0: database.get_bb("N10"), 1: database.get_bb("N409")}
    edge_bbs = {(0, 1): database.get_bb("E41")}
    return builder.build_by_type(
        topology=tbo, node_bbs=node_bbs, edge_bbs=edge_bbs
    )


def test_edge_build_atom_count(hkust1_with_edge):
    assert len(hkust1_with_edge.atoms) == 1200


def test_edge_build_composition(hkust1_with_edge):
    composition = Counter(hkust1_with_edge.atoms.get_chemical_symbols())
    assert composition == Counter({"C": 864, "O": 192, "H": 96, "Cu": 48})


def test_edge_build_adds_atoms_versus_no_edge(hkust1_with_edge):
    # Inserting E41 on every (0,1) edge must increase the atom count
    # relative to the node-only HKUST-1 (624 atoms, a README constant).
    assert len(hkust1_with_edge.atoms) > 624
