import pytest

from tests.helpers import make_clotho


@pytest.fixture
def clotho(tmp_path):
    return make_clotho(tmp_path / "clotho")
