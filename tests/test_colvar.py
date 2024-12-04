from af2rave import Colvar
import pytest
import numpy as np
import os

DAT_LENGTH = 1001
DAT_COL = 428


def check_reproduce(colvar):
    colvar.write("./tests/test_colvars/test.dat")
    new_colvar = Colvar.from_file("./tests/test_colvars/test.dat")
    assert new_colvar.header == colvar.header
    assert new_colvar.shape == colvar.shape
    assert np.allclose(new_colvar._data, colvar._data)
    os.remove("./tests/test_colvars/test.dat")


def test_import():
    assert Colvar is not None


def test_load():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    assert colvar is not None
    assert colvar._data.shape == (DAT_COL, DAT_LENGTH)


def test_missing_metadata():
    with pytest.raises(ValueError):
        colvar = Colvar.from_file("./tests/test_colvars/missing_meta_data.dat")
        assert colvar is None


def test_shape():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    assert colvar.shape == (DAT_COL, DAT_LENGTH)
    assert colvar.shape == (len(colvar.header), len(colvar._data[0]))
    assert colvar.shape == colvar._data.shape


def test_header():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    assert colvar.header == colvar._header
    assert colvar.header[0:3] == ["d5_1664", "d5_1665", "d5_1666"]
    assert len(colvar.header) == DAT_COL


def test_write():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    check_reproduce(colvar)

def test_stride():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar.stride(2)
    assert colvar.shape == (DAT_COL, np.ceil(DAT_LENGTH / 2).astype(int))
    check_reproduce(colvar)

def test_stride_no_time():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar._time = None
    colvar.stride(2)
    assert colvar.shape == (DAT_COL, np.ceil(DAT_LENGTH / 2).astype(int))
    check_reproduce(colvar)


def test_choose():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar2 = colvar.choose(["d5_1664", "d5_1665", "d5_1666"])
    assert colvar2.header == ["d5_1664", "d5_1665", "d5_1666"]
    assert colvar2.shape == (3, DAT_LENGTH)
    check_reproduce(colvar2)


def test_choose_invariance():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar2 = colvar.choose(["d5_1664", "d5_1665", "d5_1666"])
    del colvar["d5_1664"]
    assert colvar2["d5_1664"] is not None


def test_choose_invariance2():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar2 = colvar.choose(["d5_1664", "d5_1665", "d5_1666"])
    colvar["d5_1664"] = np.ones(DAT_LENGTH)
    assert not np.allclose(colvar2["d5_1664"], np.ones(DAT_LENGTH))


def test_tappend():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar.tappend(colvar)
    assert colvar.shape == (DAT_COL, 2 * DAT_LENGTH)
    check_reproduce(colvar)


def test_tappend_with_shorter():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar2 = colvar.choose(["d5_1664", "d5_1665", "d5_1666"])
    with pytest.raises(ValueError):
        colvar.tappend(colvar2)


def test_tappend_with_longer():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar2 = colvar.choose(["d5_1664", "d5_1665", "d5_1666"])
    colvar2.tappend(colvar)
    assert colvar2.shape == (3, 2 * DAT_LENGTH)
    check_reproduce(colvar2)

def test_tappend_from_nothing():
    colvar = Colvar()
    colvar2 = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar.tappend(colvar2)
    assert colvar.shape == (DAT_COL, DAT_LENGTH)
    assert np.allclose(colvar._data, colvar2._data)
    assert colvar.header == colvar2.header
    check_reproduce(colvar)

def test_tappend_no_time():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar2 = colvar.choose(["d5_1664", "d5_1665", "d5_1666"])
    colvar2._time = None
    colvar2.tappend(colvar2)
    assert colvar2.shape == (3, 2 * DAT_LENGTH)
    check_reproduce(colvar2)

def test_tappend_from_file():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar.tappend("./tests/test_colvars/base.dat")
    assert colvar.shape == (DAT_COL, 2 * DAT_LENGTH)


def test_self_tappend():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar.tappend(colvar)
    assert colvar.shape == (DAT_COL, 2 * DAT_LENGTH)
    check_reproduce(colvar)


def test_tappend_return_val():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar2 = colvar.tappend(colvar)
    assert colvar2 is not None
    assert colvar2.shape == (DAT_COL, 2 * DAT_LENGTH)
    assert np.allclose(colvar._data, colvar2._data)
    check_reproduce(colvar2)


def test_tappend_return_val_when_empty():
    colvar = Colvar()
    colvar2 = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar3 = colvar.tappend(colvar2)
    assert colvar3 is not None
    assert colvar3.shape == (DAT_COL, DAT_LENGTH)
    assert np.allclose(colvar3.data, colvar2.data)
    assert np.allclose(colvar.data, colvar3.data)
    check_reproduce(colvar2)


def test_daisy_chained_tappend():
    colvar = Colvar()
    colvar2 = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar.tappend(colvar2).tappend(colvar2)
    assert colvar.shape == (DAT_COL, 2 * DAT_LENGTH)
    assert colvar.header == colvar2.header
    check_reproduce(colvar)


def test_kappend():
    pass


def test_magic_contains():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    assert "d5_1664" in colvar
    assert "pratyush" not in colvar


def test_magic_getitem():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    assert np.allclose(colvar["d5_1664"], colvar._data[0])
    with pytest.raises(KeyError):
        colvar["pratyush"]


def test_magic_setitem():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar["pratyush"] = np.ones(DAT_LENGTH)
    assert "pratyush" in colvar
    assert np.allclose(colvar["pratyush"], np.ones(DAT_LENGTH))
    assert colvar.shape == (DAT_COL + 1, DAT_LENGTH)
    check_reproduce(colvar)


def test_magic_setitem_existing():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    colvar["d5_1664"] = np.ones(DAT_LENGTH)
    assert np.allclose(colvar["d5_1664"], np.ones(DAT_LENGTH))
    assert colvar.shape == (DAT_COL, DAT_LENGTH)
    check_reproduce(colvar)


def test_magic_setitem_wrong_length():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    with pytest.raises(ValueError):
        colvar["pratyush"] = np.ones(DAT_LENGTH - 1)


def test_magic_setitem_from_empty():
    colvar = Colvar()
    colvar["pratyush"] = np.ones(DAT_LENGTH)
    assert "pratyush" in colvar
    assert np.allclose(colvar["pratyush"], np.ones(DAT_LENGTH))
    assert colvar.shape == (1, DAT_LENGTH)
    check_reproduce(colvar)


def test_magic_delitem():
    colvar = Colvar.from_file("./tests/test_colvars/base.dat")
    del colvar["d5_1664"]
    assert "d5_1664" not in colvar
    assert colvar.shape == (DAT_COL - 1, DAT_LENGTH)
    check_reproduce(colvar)
