import pytest

import pormake


# (name, n_slots, n_nodes, n_edges, n_node_types, n_edge_types, spacegroup)
TOPOLOGY_FACTS = [
    ("tbo", 152, 56, 96, 2, 1, "Fm-3m"),
    ("pcu", 4, 1, 3, 1, 1, "Pm-3m"),
    ("acs", 8, 2, 6, 1, 1, "P63/mmc"),
    ("ith", 32, 8, 24, 2, 1, "Pm-3n"),
]


def make_topology(database, name):
    # Build directly from the cgd path to avoid the .pickle side effect
    # that database.get_topo writes into the package directory.
    return pormake.Topology(database.topo_dir / f"{name}.cgd")


@pytest.mark.parametrize(
    "name, n_slots, n_nodes, n_edges, n_node_types, n_edge_types, sg",
    TOPOLOGY_FACTS,
)
def test_topology_metadata(
    database, name, n_slots, n_nodes, n_edges, n_node_types, n_edge_types, sg
):
    topo = make_topology(database, name)
    assert topo.n_slots == n_slots
    assert topo.n_nodes == n_nodes
    assert topo.n_edges == n_edges
    assert topo.n_node_types == n_node_types
    assert topo.n_edge_types == n_edge_types
    assert topo.spacegroup == sg


@pytest.mark.parametrize("name", [f[0] for f in TOPOLOGY_FACTS])
def test_topology_index_consistency(database, name):
    topo = make_topology(database, name)
    # nodes + edges partition all slots
    assert topo.n_nodes + topo.n_edges == topo.n_slots
    assert len(topo.node_indices) == topo.n_nodes
    assert len(topo.edge_indices) == topo.n_edges
    # unique type counts match the reported counts
    assert len(topo.unique_node_types) == topo.n_node_types
    assert len(topo.unique_edge_types) == topo.n_edge_types
    # one local structure per unique node type
    assert len(topo.unique_local_structures) == topo.n_node_types
