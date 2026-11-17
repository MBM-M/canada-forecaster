import os

# Small forests keep the suite fast; must be set before the service package is imported.
os.environ.setdefault("SERVICE_RF_TREES", "40")

import pytest  # noqa: E402

from service.cli import replay  # noqa: E402
from service.data import load_raw  # noqa: E402
from service.store import connect  # noqa: E402
from tests.helpers import drain  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return load_raw()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "service.db"


@pytest.fixture
def conn(db_path):
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture(scope="session")
def finished_db(tmp_path_factory, raw):
    """One full replay processed to completion, shared by the read-only tests."""
    path = tmp_path_factory.mktemp("done") / "service.db"
    c = connect(path)
    replay(c, raw=raw)
    drain(c, raw)
    yield c
    c.close()
