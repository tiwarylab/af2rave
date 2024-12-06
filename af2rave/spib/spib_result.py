'''
Container class for SPIB results.
'''

from __future__ import annotations

import numpy as np
import pickle
from pathlib import Path
from ..colvar import Colvar

from numpy.typing import NDArray


class SPIBResult():

    def __init__(self, prefix: str, postfix: str, n_traj: int, **kwargs):
        '''
        This object should be treated as read-only to the user. 
        Not supposed to be constructed by the user. Use SPIBResult.from_file() instead.

        The method SPIBProcess.run() will be responsible for creating this container class.
        '''

        self._prefix = prefix
        self._postfix = postfix
        self._n_traj = n_traj
        self._dt = kwargs.get("dt", None)
        self._b = kwargs.get("b", 0)
        self._k = kwargs.get("k", 1)

        self._linear_interpolator = None

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

        # remove stale state labels
        self._n_input_labels = self._traj[0]["labels"].shape[1]
        self._converged_states = self._get_converged_states()
        state_idx = np.where(self._converged_states)[0]
        for i in range(self._n_traj):
            self._traj[i]["labels"] = self._traj[i]["labels"][:, state_idx]

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
        '''
        Load a numpy file with the given keyword. Provide i_traj if that file is trajectory-specific.
        '''
        if i_traj is None:
            return np.load(f"{self._prefix}_{keyword}{self._postfix}")
        else:
            return np.load(f"{self._prefix}_traj{i_traj}_{keyword}{self._postfix}")

    @classmethod
    def from_file(cls, filename: str) -> SPIBResult:
        '''
        Load a SPIBResult object from a binary pickle file.

        :param filename: The filename to load.
        :type filename: str
        :return: The SPIBResult object.
        :rtype: SPIBResult
        :raises FileNotFoundError: If the file does not exist.
        '''
        if not Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist.")
        return pickle.load(open(filename, "rb"))

    def to_file(self, filename: str) -> None:
        '''
        Dump the object into a binary pickle file for future use.

        :param filename: The filename to save.
        :type filename: str
        '''
        pickle.dump(self, open(filename, "wb"))

    @property
    def dt(self) -> float:
        '''
        Time lag for this run.
        '''
        return self._dt

    @property
    def n_traj(self) -> int:
        '''
        The number of input trajectories.
        '''
        return self._n_traj

    @property
    def n_input_labels(self) -> int:
        '''
        The number of initial states/input labels.
        Typically this is 2 * n_traj.
        '''
        return self._n_input_labels

    @property
    def n_converged_states(self) -> int:
        '''
        The number of remaining converged states.
        '''
        return np.sum(self._converged_states)

    def _get_converged_states(self) -> NDArray:
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

        p = (X - self._b.reshape(-1, 1)) / self._k.reshape(-1, 1)
        p = np.dot(self._z_mean_encoder["weight"], p) + self._z_mean_encoder["bias"].reshape(-1, 1)
        return p

    def project_colvar(self, X: Colvar) -> Colvar:
        '''
        Project the input colvar into the latent space.

        :param X: The input colvar to project.
        :type X: Colvar
        :return: The projected colvar.
        :rtype: Colvar
        '''
        return X.map(self.project, insitu=False)

    def get_latent_representation(self, traj_idx: list[int] | int = None) -> NDArray:
        '''
        Return the latent representation of the trajectory.
        If no index is provides, return all trajectories.
        '''

        if traj_idx is not None:
            idx = np.atleast_1d(traj_idx)
            rep = np.vstack([self._traj[i]["mean_representation"] for i in idx])
        else:
            rep = np.vstack([traj["mean_representation"] for traj in self._traj])
        return rep.T

    def get_state_label(self, traj_idx: int = None):
        '''
        Return the state label of the trajectory.
        If no index is provides, return all trajectories.

        :param traj_idx: The index of the trajectory. If None, return all trajectories.
        :type traj_idx: int, optional
        :return: The state label as a number from 0 to n_states. shape: n_frames
        :rtype: np.ndarray
        '''

        if traj_idx is not None:
            idx = np.atleast_1d(traj_idx)
            state = np.hstack([self._traj[i]["labels"].nonzero()[1] for i in idx])
        else:
            state = np.hstack([traj["labels"].nonzero()[1] for traj in self._traj])
        return state

    def get_traj_label(self, traj_idx: int = None):
        '''
        Return the trajectory label of the trajectory, 
        i.e. the index of the trajectory each frame belongs to.
        If no index is provides, return all trajectories.
        When a index is provided, the return value the same as [index] * nframes.

        :param traj_idx: The index of the trajectory. If None, return all trajectories.
        :type traj_idx: int, optional
        :return: The trajectory label. shape: n_frames
        :rtype: np.ndarray
        '''

        if traj_idx is not None:
            idx = np.atleast_1d(traj_idx)
            traj = np.hstack([np.full(self._traj[i]["labels"].shape[0], i) for i in idx])
        else:
            traj = np.hstack([np.full(traj["labels"].shape[0], i) for i, traj in enumerate(self._traj)])
        return traj

    def get_probability_distribution(self, nbins=200) -> tuple[NDArray, NDArray, NDArray]:
        '''
        Get the probability distribution in the latent space.

        :param nbins: The number of bins for the histogram. The format is the same with np.histogram2d.
        :type nbins: int, (int, int), optional, default=200
        :return: x, y, h
        '''

        h, x, y = np.histogram2d(*self.get_latent_representation(), bins=nbins, density=True)
        return x, y, h.T    # what the hell is this transpose?

    def get_free_energy(self, nbins=200) -> tuple[NDArray, NDArray, NDArray]:
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
    
    def project_state_label(self, X: NDArray):
        '''
        Project an arbitary coordinate in the latent space to the most probable state.
        NaN will be returned if the coordinate is outside the convex hull of the latent space.

        This algorithm interpolates a one-hot vector representation of state labels 
        over the latent space with a 2D Clough-Tocher interpolator as implemented in scipy.

        :param X: The coordinate, shape: 2 * npoints
        :type X: np.ndarray
        :return: The state label.
        :rtype: np.ndarray
        '''

        from scipy.interpolate import CloughTocher2DInterpolator

        def nan_argmax(array: NDArray) -> NDArray:
            '''
            This function behaves like np.argmax(array, axis=1), but returns NaN if any element in the row is NaN.
            '''
            nan_mask = np.isnan(array).any(axis=1)
            
            # Use argmax along axis=1 for rows without NaN
            max_indices = np.argmax(np.where(nan_mask[:, None], -np.inf, array), axis=1)
            
            # Replace indices with NaN where NaN was present in the row
            max_indices = max_indices.astype(float)  # Convert to float for NaN compatibility
            max_indices[nan_mask] = np.nan
            
            return max_indices
        
        if X.shape[0] != 2:
            raise ValueError("The input shape is not compatible with the latent space. Shape of the input need to be N * 2.")

        if getattr(self, "_interpolator", None) is None:
            x = self.get_latent_representation().T
            y = np.eye(self.n_converged_states)[self.get_state_label()]
            self._interpolator = CloughTocher2DInterpolator(x, y)
        
        onehot_y = self._interpolator(X.T)
        return nan_argmax(onehot_y)