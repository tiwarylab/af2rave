from af2rave.feature.colvar import Colvar
import af2rave.amino as amino
import pytest
import numpy as np
import os


def test_import():
    assert amino is not None
    assert amino.AMINO is not None


def test_from_colvar():
    a = amino.AMINO()
    a.from_colvar("./tests/test_colvars/base.dat")
    assert a.result is not None
    assert a.get_colvar() is not None
    a.write_colvar("./tests/test_colvars/test.dat")

    new_colvar = Colvar("./tests/test_colvars/test.dat")
    # ["d1412_1665", "d1182_1666"]
    assert new_colvar.header == a.get_colvar().header
    assert new_colvar.shape == a.get_colvar().shape
    assert np.allclose(new_colvar._data, a.get_colvar()._data)
    os.remove("./tests/test_colvars/test.dat")


def test_no_run_get_colvar():
    a = amino.AMINO()
    with pytest.raises(ValueError):
        a.get_colvar()
