import pytest

import pormake


@pytest.fixture(scope="session")
def database():
    return pormake.Database()


@pytest.fixture(scope="session")
def builder():
    return pormake.Builder()


@pytest.fixture(scope="session")
def locator():
    return pormake.Locator()
