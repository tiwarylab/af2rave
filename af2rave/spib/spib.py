'''
The SPIB data analysis module.
'''

import torch
import numpy as np
import time
import hashlib

from .spib_result import SPIBResult
from .wrapper import spib as spib_kernel

class SPIBProcess(object):

    def __init__(self, 
                 traj: str | list[str], **kwargs):

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if isinstance(traj, str):
            traj = [traj]
        self._n_traj = len(traj)
        self._kwargs = kwargs

        # load and store the trajectory data
        # ---------------------------------------------
        self._traj_data_list = []
        self._traj_labels_list = []

        n_states = self._n_traj * 2

        self._min = np.inf
        self._max = -np.inf

        for i, f in enumerate(traj):

            data = np.loadtxt(f)
            n_data = data.shape[0]

            self._min = np.minimum(self._min, np.min(data, axis=0))
            self._max = np.maximum(self._max, np.max(data, axis=0))

            scalar_label = np.rint(np.arange(n_data) >= n_data/2).astype(int) + i * 2
            onehot_label = np.eye(n_states)[scalar_label]

            data = torch.tensor(data, dtype=torch.float32).to(self._device)
            label = torch.tensor(onehot_label, dtype=torch.float32).to(self._device)

            self._traj_data_list.append(data)
            self._traj_labels_list.append(label)
        
        for i in range(len(self._traj_data_list)):
            self._traj_data_list[i] = (self._traj_data_list[i] - self._min) / (self._max - self._min)
        # ---------------------------------------------

    def run(self, time_lag: int, **kwargs):

        basename = "tmp_" + hashlib.md5(str(time.time()).encode()).hexdigest()
        seed = self._kwargs.get('seed', 42)
        
        spib_kernel(self._traj_data_list, self._traj_labels_list, time_lag, 
                    base_path=basename, device=self._device,
                    **kwargs)
        
        prefix = f"{basename}/model_dt_{time_lag}"
        postfix = f"{seed}.npy"

        return SPIBResult(prefix, postfix, self._n_traj, 
                          dt=time_lag, min=self._min, max=self._max)
    
    