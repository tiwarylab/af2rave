from af2rave.colvar import Colvar
import af2rave.amino as amino
import pytest

from tests.test_colvar import check_reproduce

def test_import():
    assert amino is not None
    assert amino.AMINO is not None

class TestInterface:

    a = amino.AMINO.from_file("./tests/test_colvars/base.dat")

    def test_result(self):
        assert self.a.result is not None
        assert len(self.a.result) > 0

    def test_to_colvar(self):
        assert self.a.to_colvar() is not None
        assert self.a.to_colvar().shape[0] == len(self.a.result)

    def test_integrity_colvar(self):
        check_reproduce(self.a.to_colvar())

    b = amino.AMINO()
    def test_no_run_get_colvar(self):
        with pytest.raises(ValueError):
            self.b.to_colvar()