'''
The SPIB data analysis module.
'''

import torch
import numpy as np
import time
import hashlib

from .spib_result import SPIBResult
from .wrapper import spib as spib_kernel
from ..colvar import Colvar


class SPIBProcess(object):

    def __init__(self,
                 traj: str | list[str], **kwargs):

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if isinstance(traj, str):
            self._traj = [traj]
        else:
            self._traj = traj
        self._n_traj = len(traj)
        self._kwargs = kwargs

        self._traj_data_list = self._load_data()
        self._traj_labels_list = self._init_default_label()
        self._min_max_scaling()

    def _load_data(self) -> list[torch.Tensor]:
        '''
        Load the colvar data from the trajectory files.

        :return traj_data_list: list of torch.Tensor in designated device
        :rtype: list[torch.Tensor]
        '''

        traj_data_list = []

        for f in self._traj:
            data = Colvar.from_file(f).data.T
            data = torch.tensor(data, dtype=torch.float32).to(self._device)
            self._traj_data_list.append(data)
        
        return traj_data_list

    def _init_default_label(self):
        '''
        Initialize the default labels for the trajectories.
        
        The default labels are one-hot encoded, with the first half of the trajectory
        labeled as 0 and the second half labeled as 1.

        :return traj_labels_list: list of torch.Tensor in designated device
        :rtype: list[torch.Tensor]
        '''

        traj_labels_list = []

        for i, f in enumerate(self._traj_data_list):
            n_data = f.shape[0]

            scalar_label = np.rint(np.arange(n_data) >= n_data / 2).astype(int) + i * 2
            onehot_label = np.eye(self._n_traj * 2)[scalar_label]
            label = torch.tensor(onehot_label, dtype=torch.float32).to(self._device)

            traj_labels_list.append(label)
        
        return traj_labels_list

    def _min_max_scaling(self):

        self._min, self._max = np.inf, -np.inf

        for i, t in enumerate(self._traj_data_list):
            self._min = np.minimum(self._min, torch.min(t, 0).values.numpy(), dtype=np.float32)
            self._max = np.maximum(self._max, torch.max(t, 0).values.numpy(), dtype=np.float32)

        self._b = self._min
        self._k = self._max - self._min

        for i in range(self._n_traj):
            self._traj_data_list[i] = (self._traj_data_list[i] - self._b) / self._k

    def run(self, time_lag: int, **kwargs):

        basename = "tmp_" + hashlib.md5(str(time.time()).encode()).hexdigest()
        seed = self._kwargs.get('seed', 42)

        spib_kernel(self._traj_data_list, self._traj_labels_list, [time_lag],
                    base_path=basename, device=self._device,
                    **kwargs)

        prefix = f"{basename}/model_dt_{time_lag}"
        postfix = f"{seed}.npy"

        return SPIBResult(prefix, postfix, self._n_traj,
                          dt=time_lag, b=self._b, k=self._k)
