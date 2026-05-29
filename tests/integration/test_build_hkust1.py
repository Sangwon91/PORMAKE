from collections import Counter

import pytest


@pytest.fixture(scope="module")
def hkust1(database, builder):
    tbo = database.get_topo("tbo")
    node_bbs = {0: database.get_bb("N10"), 1: database.get_bb("N409")}
    return builder.build_by_type(topology=tbo, node_bbs=node_bbs)


def test_hkust1_atom_count(hkust1):
    assert len(hkust1.atoms) == 624


def test_hkust1_composition(hkust1):
    composition = Counter(hkust1.atoms.get_chemical_symbols())
    assert composition == Counter({"C": 288, "O": 192, "H": 96, "Cu": 48})


def test_hkust1_cell_lengths_are_sane(hkust1):
    a, b, c = hkust1.atoms.cell.cellpar()[:3]
    # Cubic HKUST-1; lengths captured at ~27.14 on the baseline platform.
    # Loose rel tolerance absorbs cross-platform jax/BLAS differences.
    assert a == pytest.approx(27.14, rel=5e-2)
    assert b == pytest.approx(27.14, rel=5e-2)
    assert c == pytest.approx(27.14, rel=5e-2)
