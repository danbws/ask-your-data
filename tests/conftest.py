import pytest

from ask_your_data.db import SQLiteDatabase
from ask_your_data.sample_data import build_sample_db


@pytest.fixture(scope="session")
def sample_db_path(tmp_path_factory):
    return build_sample_db(tmp_path_factory.mktemp("data") / "sample.db")


@pytest.fixture()
def db(sample_db_path):
    return SQLiteDatabase(sample_db_path)
