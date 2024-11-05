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
    def n_traj(self):
        return self._n_traj

    @property
    def n_input_labels(self):
        return self._traj[0]["labels"].shape[1]

    @property
    def n_converged_states(self) -> int:
        '''
        The number of remaining converged states.
        '''
        return np.sum(self._converged_states)

    def _get_converged_states(self) -> np.ndarray:
        '''
        Return a one-hot encoding of all remaining states.

        :return label: The one-hot encoding of the remaining states. Dimension: n_input_labels
        :rtype: np.ndarray
        '''

        labels = np.zeros(self.n_input_labels, dtype=int)
        for i in np.arange(self._n_traj):
            labels |= np.any(self._traj[i]["labels"], axis=0)
        return labels

    def project(self, X):
        '''
        Project the input data to the latent space.

        :param X: The input data to project. Dimension: n_input_dims x n_frames
        :type X: np.ndarray
        '''

        # shape of X: n_input_dims x n_frames
        # shape of weight: 2 x n_input_dims
        # shape of bias: 2

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

    def get_latent_representation(self, traj_idx: int = None):
        '''
        Return the latent representation of the trajectory.
        If no index is provides, return all trajectories.
        '''

        if traj_idx is not None:
            rep = self._traj[traj_idx]["representation"]
        else:
            rep = np.vstack([traj["representation"] for traj in self._traj])
        return rep.T

    def get_state_label(self, traj_idx: int = None):
        '''
        Return the state label of the trajectory.
        If no index is provides, return all trajectories.
        '''

        if traj_idx is not None:
            state = self._traj[traj_idx]["labels"].nonzero()[1]
        else:
            state = np.hstack([traj["labels"].nonzero()[1] for traj in self._traj])
        return state

    def get_traj_label(self, traj_idx: int = None):
        '''
        Return the trajectory label of the trajectory.
        If no index is provides, return all trajectories.
        '''

        if traj_idx is not None:
            traj = np.full(self._traj[traj_idx]["labels"].shape[0], traj_idx)
        else:
            traj = np.hstack([np.full(traj["labels"].shape[0], i) for i, traj in enumerate(self._traj)])
        return traj

    def get_probability_distribution(self, nbins=200):

        h, x, y = np.histogram2d(*self.get_latent_representation(), bins=nbins, density=True)
        return x, y, h

    def get_free_energy(self, nbins=200):
        '''
        Get the free energy as the negative logarithm of the probability distribution. Unit: kT

        :param nbins: The number of bins for the histogram. The format is the same with np.histogram2d.
        :type nbins: int, (int, int), optional, default=200
        :return: x, y, f

        Example:
        - Plot in matplotlib

        `plt.pcolor(*result.get_free_energy(), cmap="RdBu_r", shading="auto")`

        The sequence of the return value allows a direct input to the pcolor function.
        '''

        x, y, h = self.get_probability_distribution(nbins)
        f = -np.log(h)
        f -= np.min(f)
        return x, y, f
