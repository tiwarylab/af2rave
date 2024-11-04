'''
Container class for SPIB results.
'''

import numpy as np
import pickle
from ..colvar import Colvar

class SPIBResult():

    def __init__(self, prefix: str, postfix: str, n_traj: int, **kwargs):

        self._prefix = prefix
        self._postfix = postfix
        self._n_traj = n_traj
        self._dt = kwargs.get("dt", None)
        self._min = kwargs.get("min", None)
        self._max = kwargs.get("max", None)

        # per trajectory data
        self._traj = [{} for _ in range(self._n_traj)]
        for i in range(self._n_traj):

            # n_frames x n_input_labels
            self._traj[i]["data_prediction"] = self._np_load("data_prediction", i)
            self._traj[i]["labels"] = self._np_load("labels", i)

            # 1 x n_input_labels
            self._traj[i]["state_population"] = self._np_load("state_population", i)      
            
            # n_frames x RC_dim
            self._traj[i]["mean_representation"] = self._np_load("mean_representation", i)
            self._traj[i]["representation"] = self._np_load("representation", i)

        self._converged_states = self._get_converged_states()

        # encoder params
        self._z_mean_encoder = {}
        self._z_mean_encoder["bias"] = self._np_load("z_mean_encoder_bias")
        self._z_mean_encoder["weight"] = self._np_load("z_mean_encoder_weight")

        # representative input
        self._representatives = {}
        self._representatives["inputs"] = self._np_load("representative_inputs")
        self._representatives["weight"] = self._np_load("representative_weight")
        self._representatives["z_logvar"] = self._np_load("representative_z_logvar")
        self._representatives["z_mean"] = self._np_load("representative_z_mean")

    def _np_load(self, keyword: str, i_traj: int = None):
        if i_traj is None:
            return np.load(f"{self._prefix}_{keyword}{self._postfix}")
        else:
            return np.load(f"{self._prefix}_traj{i_traj}_{keyword}{self._postfix}")

    @classmethod
    def from_file(cls, filename: str):
        return pickle.load(open(filename, "rb"))

    def to_file(self, filename: str):
        pickle.dump(self, open(filename, "wb"))

    @property
    def dt(self):
        return self._dt
    
    @property
    def n_input_labels(self):
        return self._traj[0]["labels"].shape[1]
    
    @property
    def n_converged_states(self):
        return np.sum(self._converged_states)
    
    def _get_converged_states(self):
        labels = np.zeros(self.n_input_labels, dtype=int)
        for i in np.arange(self._n_traj):
            labels |= np.any(self._traj[i]["labels"], axis=0)
        return labels
    
    def project(self, X):

        # shape of X: n_input_dims x n_frames
        # shape of weight: RC_dim x n_input_dims
        # shape of bias: RC_dim
        
        min_max_scaling = (self._min is not None) and (self._max is not None)
        if min_max_scaling:
            b, k = self._min, self._max - self._min
        else:
            b, k = 0, 1
            print("[SPIBResult.project] Missing min-max scaling information.")
        p = (X - b) / k
        p = np.dot(self._z_mean_encoder["weight"], p) + self._z_mean_encoder["bias"]
        return p
        
    def project_colvar(self, X: Colvar):

        min_max_scaling = (self._min is not None) and (self._max is not None)
        if min_max_scaling:
            b, k = self._min, self._max - self._min
        else:
            b, k = 0, 1
            print("[SPIBResult.project_colvar] Missing min-max scaling information.")

        scaling = lambda x: (x - b) / k
        Z = X.map(scaling, insitu=False)
        projection = lambda x: np.dot(self._z_mean_encoder["weight"], x) + self._z_mean_encoder["bias"]
        Z.map(projection)

        return Z

