import pytest

import pormake


def test_get_bb_returns_building_block(database):
    bb = database.get_bb("N10")
    assert isinstance(bb, pormake.BuildingBlock)
    assert bb.name == "N10"


def test_get_topo_returns_topology(database):
    topo = database.get_topo("pcu")
    assert isinstance(topo, pormake.Topology)
    assert topo.name == "pcu"


def test_get_bb_missing_name_raises(database):
    with pytest.raises(Exception):
        database.get_bb("definitely_not_a_real_bb_name_zzz")


def test_get_topo_missing_name_raises(database):
    with pytest.raises(Exception):
        database.get_topo("definitely_not_a_real_topo_zzz")


def test_bb_list_nonempty_and_contains_known(database):
    names = database.bb_list
    assert len(names) > 0
    assert "N10" in names
    assert "E41" in names


def test_topo_list_nonempty_and_contains_known(database):
    names = database.topo_list
    assert len(names) > 0
    assert "tbo" in names
    assert "pcu" in names
