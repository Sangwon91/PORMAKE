from collections import Counter

import pytest


@pytest.fixture(scope="module")
def chimera_mof(database, builder):
    tbo = database.get_topo("tbo")
    node_bbs = {0: database.get_bb("N10"), 1: database.get_bb("N409")}
    edge_bbs = {(0, 1): database.get_bb("E41")}
    bbs = builder.make_bbs_by_type(
        topology=tbo, node_bbs=node_bbs, edge_bbs=edge_bbs
    )
    n13 = database.get_bb("N13")
    for idx in [33, 38, 40, 49, 53, 55]:
        bbs[idx] = n13.copy()
    return builder.build(topology=tbo, bbs=bbs)


def test_chimera_atom_count(chimera_mof):
    assert len(chimera_mof.atoms) == 1320


def test_chimera_composition(chimera_mof):
    composition = Counter(chimera_mof.atoms.get_chemical_symbols())
    assert composition == Counter(
        {"C": 960, "H": 156, "O": 144, "Cu": 36, "N": 24}
    )


def test_chimera_contains_nitrogen_from_porphyrin(chimera_mof):
    # N13 (porphyrin) introduces nitrogen, which pure HKUST-1 lacks.
    symbols = set(chimera_mof.atoms.get_chemical_symbols())
    assert "N" in symbols
