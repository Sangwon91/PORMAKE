from collections import Counter

import ase.io
import pytest

import pormake


@pytest.fixture(scope="module")
def hkust1(database, builder):
    tbo = database.get_topo("tbo")
    node_bbs = {0: database.get_bb("N10"), 1: database.get_bb("N409")}
    return builder.build_by_type(topology=tbo, node_bbs=node_bbs)


def test_cif_roundtrip_preserves_atom_count(hkust1, tmp_path):
    cif = tmp_path / "hkust1.cif"
    hkust1.write_cif(str(cif))
    reloaded = ase.io.read(str(cif))
    assert len(reloaded) == len(hkust1.atoms)
    assert len(reloaded) == 624


def test_cif_roundtrip_preserves_composition(hkust1, tmp_path):
    cif = tmp_path / "hkust1.cif"
    hkust1.write_cif(str(cif))
    reloaded = ase.io.read(str(cif))
    assert Counter(reloaded.get_chemical_symbols()) == Counter(
        hkust1.atoms.get_chemical_symbols()
    )


def test_cif_file_is_created(hkust1, tmp_path):
    cif = tmp_path / "hkust1.cif"
    hkust1.write_cif(str(cif))
    assert cif.exists()
    assert cif.stat().st_size > 0
