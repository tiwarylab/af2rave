from af2rave import feature
from glob import glob
import numpy as np
import pytest

pdb_dir = "tests/test_pdb/"
ref_pdb = f"{pdb_dir}/pred_1.pdb"
pdb_files = glob(f"{pdb_dir}/*.pdb")

def test_import():
    assert feature is not None
    assert feature.FeatureSelection is not None

class TestInit:

    def test_init_no_ref(self):
        fs = feature.FeatureSelection(pdb_files)
        assert fs.traj.n_frames == 111
        assert fs._ref_pdb is not None

    def test_init_with_ref(self):
        fs = feature.FeatureSelection(pdb_files, ref_pdb)
        assert fs.traj.n_frames == 111
        assert fs._ref_pdb is not None
        assert fs._traj is not None
        assert fs._ref is not None

    def test_init_with_dir(self):
        fs = feature.FeatureSelection(pdb_dir)
        assert fs._ref_pdb.endswith("pred_1.pdb")

    def test_init_with_dir_and_files(self):
        fs_dir = feature.FeatureSelection(pdb_dir, pdb_files)
        fs_files = feature.FeatureSelection(pdb_files)
        for i in fs_dir.pdb_name:
            assert i in fs_files.pdb_name
        
    def test_init_with_non_existent_dir(self):
        with pytest.raises(FileNotFoundError):
            feature.FeatureSelection("non_existent_dir")


class TestProperties:
    '''
    This module tests the functions in feature.py, the feature class
    '''

    fs = feature.FeatureSelection(pdb_dir, ref_pdb)

    def test_len(self):
        assert len(self.fs) == 111

    def test_get_item(self):
        for p in pdb_files:
            assert self.fs[p] is not None

    def test_empty_feature_array(self):
        assert self.fs.feature_array.shape == (0, 0)
        self.fs.rank_feature("name CA")
        assert self.fs.feature_array.shape[0] == 111
    

class TestFilters:

    fs = feature.FeatureSelection(pdb_dir, ref_pdb)

    def test_rmsd_filter(self):

        rmsd = {
            0.0: 1,
            1.0: 1,
            2.0: 29,
            3.0: 62,
            4.0: 73,
            10.0: 111
        }

        for query, count in rmsd.items():
            assert len(self.fs.rmsd_filter("name CA", query)) == count

        # Test error throwing
        with pytest.raises(ValueError):
            self.fs.rmsd_filter("name CA", -1.0)
    
    def test_steric_clash_filter(self):
        
        min_nb_dist = {
            -1.0: 111,
            0.1: 111,
            0.5: 105,
            1.2: 43
        }

        for query, count in min_nb_dist.items():
            assert len(self.fs.steric_clash_filter(query)) == count
        
        # Test error throwing. No strucuture should satisfy this
        with pytest.raises(ValueError):
            self.fs.steric_clash_filter(3)
    
    def test_pb_filter(self):

        assert len(self.fs.peptide_bond_filter(1.32, 0.1)) == 68
        assert len(self.fs.peptide_bond_filter(1.33, 0.1)) == 108
        assert len(self.fs.peptide_bond_filter(1.34, 0.1)) == 110
        assert len(self.fs.peptide_bond_filter(1.35, 0.05)) == 9
        assert len(self.fs.peptide_bond_filter(1.35, 0.06)) == 92
        assert len(self.fs.peptide_bond_filter(1.35, 0.07)) == 105
        assert len(self.fs.peptide_bond_filter(1.35, 0.08)) == 107
        assert len(self.fs.peptide_bond_filter(1.35, 1.0)) == 111

        # Test error throwing
        with pytest.raises(ValueError):
            self.fs.peptide_bond_filter(1.30, 10)
        with pytest.raises(ValueError):
            self.fs.peptide_bond_filter(2.00, 0.04)

class TestApplyFilters:

    def test_apply_empty_filter(self):
        fs = feature.FeatureSelection(pdb_files[:10], ref_pdb)
        with pytest.raises(ValueError):
            fs.apply_filter([])   
    
    def test_apply_filter(self):
        fs = feature.FeatureSelection(pdb_files[:10], ref_pdb)
        fs.apply_filter([pdb_files[0], pdb_files[1], pdb_files[2]])
        assert len(fs) == 3
    
    def test_apply_multiple_filter_with_overlap(self):
        fs = feature.FeatureSelection(pdb_files[:10], ref_pdb)
        filter_1 = [pdb_files[0], pdb_files[1], pdb_files[2]]
        filter_2 = [pdb_files[0], pdb_files[3], pdb_files[4]]
        fs.apply_filter(filter_1, filter_2)
        assert len(fs) == 1

    def test_apply_multiple_filter_no_overlap(self):
        fs = feature.FeatureSelection(pdb_files[:10], ref_pdb)
        filter_1 = [pdb_files[0], pdb_files[1], pdb_files[2]]
        filter_2 = [pdb_files[3], pdb_files[4]]
        with pytest.raises(ValueError):
            fs.apply_filter(filter_1, filter_2)
