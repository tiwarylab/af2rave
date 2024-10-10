from af2rave import simulation
import pytest

pdb_file = "./datasets/DDR1_MSA8/pred_1.pdb"

def test_import():
    assert simulation is not None
    assert simulation.Charmm36mFF is not None

class TestPreparation:

    simbox = simulation.SimulationBox(pdb_file)
    simbox.create_box()

    def test_create_box(self):

        assert self.simbox is not None
        assert self.simbox.pos is not None
        assert self.simbox.top is not None 
    
    def test_mapping_int(self):
        assert self.simbox.map_atom_index(1000) == 2019
        assert self.simbox.map_atom_index(1001) == 2020
    
    def test_mapping_set(self):
        assert self.simbox.map_atom_index({1000, 1001}) == {2019, 2020}

    def test_mapping_list(self):
        assert self.simbox.map_atom_index([1000, 1001]) == [2019, 2020]
        assert self.simbox.map_atom_index([{1000, 1001}]) == [{2019, 2020}]

    def test_mapping_error(self):
        with pytest.raises(ValueError):
            assert self.simbox.map_atom_index(9999) == 9999
            assert self.simbox.map_atom_index({9999}) == {9999}
            assert self.simbox.map_atom_index([9999]) == [9999]
            assert self.simbox.map_atom_index([{9999}]) == [{9999}]

