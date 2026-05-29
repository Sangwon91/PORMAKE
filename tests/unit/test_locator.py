import pytest

import pormake


def make_topology(database, name):
    return pormake.Topology(database.topo_dir / f"{name}.cgd")


def test_rmsd_low_for_matching_node(database, locator):
    # acs node type 0 is a triangular prism; N198 is a triangular-prism
    # metal cluster, so the RMSD should be near zero.
    acs = make_topology(database, "acs")
    n198 = database.get_bb("N198")
    rmsd = locator.calculate_rmsd(acs.unique_local_structures[0], n198)
    assert rmsd == pytest.approx(0.02, abs=0.05)


def test_rmsd_high_for_mismatching_node(database, locator):
    # pcu node type 0 is an octahedron; N198 (triangular prism) fits poorly.
    pcu = make_topology(database, "pcu")
    n198 = database.get_bb("N198")
    rmsd = locator.calculate_rmsd(pcu.unique_local_structures[0], n198)
    assert rmsd == pytest.approx(0.42, rel=0.1)


def test_matching_node_has_lower_rmsd_than_mismatching(database, locator):
    n198 = database.get_bb("N198")
    acs = make_topology(database, "acs")
    pcu = make_topology(database, "pcu")
    rmsd_acs = locator.calculate_rmsd(acs.unique_local_structures[0], n198)
    rmsd_pcu = locator.calculate_rmsd(pcu.unique_local_structures[0], n198)
    assert rmsd_acs < rmsd_pcu
