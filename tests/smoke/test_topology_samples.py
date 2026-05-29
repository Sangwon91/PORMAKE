import pytest

import pormake


SAMPLE_TOPOLOGIES = [
    "tbo",
    "pcu",
    "acs",
    "ith",
    "pts",
    "dia",
    "nbo",
    "srs",
    "ths",
    "qom",
    "bcu",
    "fcu",
    "lvt",
    "soc",
    "rht",
    "bnn",
    "lon",
    "hxg",
    "hms",
    "mtn",
]


@pytest.mark.parametrize("name", SAMPLE_TOPOLOGIES)
def test_topology_loads(database, name):
    topo = database.get_topo(name)
    assert isinstance(topo, pormake.Topology)
    assert topo.n_slots > 0


@pytest.mark.parametrize("name", SAMPLE_TOPOLOGIES)
def test_topology_has_unique_local_structures(database, name):
    topo = database.get_topo(name)
    locals_ = topo.unique_local_structures
    assert len(locals_) > 0, f"Topology {name} has no unique local structures"
