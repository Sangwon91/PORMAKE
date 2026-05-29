import pytest

import pormake


SAMPLE_BBS = [
    "N3",
    "N10",
    "N13",
    "N114",
    "N198",
    "N409",
    "E41",
    "N1",
    "N100",
    "N200",
    "N300",
    "N400",
    "N500",
    "N600",
    "N700",
    "N710",
    "E1",
    "E50",
    "E110",
    "E180",
    # High coordination numbers (parse larger/denser xyz files).
    "N233",  # CN 7
    "N103",  # CN 8 (largest BB group)
    "N401",  # CN 9
    "N104",  # CN 10
    "N22",   # CN 24 (432 atoms)
]


@pytest.mark.parametrize("name", SAMPLE_BBS)
def test_bb_loads(database, name):
    bb = database.get_bb(name)
    assert isinstance(bb, pormake.BuildingBlock)
    assert bb.atoms is not None
    assert len(bb.atoms) > 0


@pytest.mark.parametrize("name", SAMPLE_BBS)
def test_bb_has_connection_points(database, name):
    bb = database.get_bb(name)
    # 'X' atoms denote connection points (per README section 6)
    n_x = sum(1 for sym in bb.atoms.get_chemical_symbols() if sym == "X")
    assert n_x > 0, f"BB {name} has no connection points (X atoms)"
