import pytest
import af2rave.spib as spib


class TestResults():

    def test_import(self):
        assert spib.SPIBResult is not None