from af2rave import feature
import os

pdb_dir = f"./tests/test_pdb/"
ref_pdb = f"{pdb_dir}/pred_1.pdb"
pdb_files = [f"{pdb_dir}/{file}" for file in os.listdir(pdb_dir) if file.endswith('.pdb')]

def test_import():
    assert feature is not None
    assert feature.Feature is not None
    assert feature.FeatureSelection is not None

def test_init_no_ref():
    fs = feature.FeatureSelection(pdb_files)
    assert fs.traj.n_frames == 111
    assert fs.ref_pdb is not None

def test_init_with_ref():
    fs = feature.FeatureSelection(pdb_files, ref_pdb)
    assert fs.traj.n_frames == 111
    assert fs.ref_pdb is not None
    assert fs.traj is not None
    assert fs.ref is not None

def test_init_with_dir():
    fs = feature.FeatureSelection(pdb_dir)
    assert fs.ref_pdb.endswith("pred_1.pdb")

class TestProperties:
    '''
    This module tests the functions in feature.py, the feature class
    '''

    fs = feature.FeatureSelection(pdb_files, ref_pdb)

    def test_len(self):
        assert len(self.fs) == 111
