import pormake


def test_top_level_import_succeeds():
    assert pormake is not None


def test_public_symbols_available():
    expected = {
        "Builder",
        "BuildingBlock",
        "Database",
        "Locator",
        "Scaler",
        "Topology",
    }
    missing = expected - set(dir(pormake))
    assert not missing, f"missing public symbols: {missing}"


def test_database_instantiates():
    db = pormake.Database()
    assert db is not None


def test_builder_instantiates():
    builder = pormake.Builder()
    assert builder is not None
