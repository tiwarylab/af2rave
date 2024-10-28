from af2rave.feature.colvar import Colvar
import pytest

def test_import():
    assert Colvar is not None

def test_load():
    colvar = Colvar("./tests/test_colvars/base.dat")
    assert colvar is not None

def test_missing_metadata():
    with pytest.raises(ValueError):
        colvar = Colvar("./tests/test_colvars/missing_meta_data.dat")
        assert colvar is None
    