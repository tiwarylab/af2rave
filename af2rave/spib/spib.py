'''
The SPIB data analysis module.
'''

import torch
import numpy as np
import time
import hashlib
import shutil

from .spib_result import SPIBResult
from .wrapper import spib as spib_kernel
from ..colvar import Colvar


class SPIBProcess(object):

    '''
    This is the af2rave wrapper for SPIB.
    To initialize, provide the list of Colvar files for SPIB to process.

    :param traj: The list of trajectory files to process.
    :type traj: str | list[str]
    :param init: 
        The way initial labels are initialized. 
        Default is "split" which split each piece of the trajectory in half.
        Available options include: "tica:n" where n is the number of clusters.
    :type init: str
    '''

    def __init__(self, traj: str | list[str], **kwargs):

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[af2rave.spib] Using device: {self._device}")

        if isinstance(traj, str):
            self._traj = [traj]
        else:
            self._traj = traj
        self._n_traj = len(traj)
        self._kwargs = kwargs
        self._init = kwargs.get('init', 'split')
        self._traj_labels_list = None
        
        # garbage collection
        self._basename = []

        self._traj_data_list = self._load_data()
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
            traj_data_list.append(data)
        
        return traj_data_list

    def _init_default_label(self, **kwargs):
        '''
        This method inits the default labels and fill in self._traj_labels_list

        There are two kinds of initial labeling: 
            - TICA + kmeans where k is the number of clusters
            - Split the trajectory in half and label the first half as 0 and the second half as 1.
        Initializing TICA requires a time lag, which is determined on run time.
        The latter can be done when initializing the trajectories.
        These will be taken care of separately.
        '''

        if self._init == 'split':
            if not self._traj_labels_list:
                self._traj_labels_list = self._init_default_label_split()
            return
        elif self._init.startswith('tica:'):
            n_clusters = int(self._init.split(':')[1])
            time_lag = kwargs.get('time_lag', None)
            self._traj_labels_list = self._init_default_label_tica(n_clusters, time_lag)
        else:
            raise ValueError(f"Unknown label initializing method: {self._init}")
            

    def _init_default_label_tica(self, n_clusters: int, time_lag: int):

        from deeptime.decomposition import TICA
        from deeptime.clustering import KMeans

        _traj_data_cpu = [d.cpu().numpy() for d in self._traj_data_list]

        tica = TICA(dim=2)
        for dd in _traj_data_cpu:
            tica.partial_fit((dd[:-time_lag], dd[time_lag:]))
        tica_model = tica.fetch_model()

        _all_coord = np.concatenate(_traj_data_cpu, axis=0)
        tica_coord = tica_model.transform(_all_coord)

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(tica_coord)
        kmeans_model = kmeans.fetch_model()

        traj_labels_list = []
        for i, dd in enumerate(_traj_data_cpu):
            scalar_label = kmeans_model.transform(tica_model.transform(dd))
            onehot_label = np.eye(n_clusters)[scalar_label]
            label = torch.tensor(onehot_label, dtype=torch.float32).to(self._device)
            traj_labels_list.append(label)

        return traj_labels_list

    def _init_default_label_split(self):
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

        self._min = torch.tensor(np.inf, dtype=torch.float32).to(self._device)
        self._max = torch.tensor(-np.inf, dtype=torch.float32).to(self._device)

        for i, t in enumerate(self._traj_data_list):
            self._min = torch.minimum(self._min, torch.min(t, 0).values)
            self._max = torch.maximum(self._max, torch.max(t, 0).values)

        self._b = self._min
        self._k = self._max - self._min

        for i in range(self._n_traj):
            self._traj_data_list[i] = (self._traj_data_list[i] - self._b) / self._k

    def run(self, time_lag: int, **kwargs):
        '''
        Run SPIB on the loaded data.

        :param time_lag: The time lag for SPIB.
        :type time_lag: int
        :return: SPIBResult object.
        :rtype: SPIBResult
        '''

        self._init_default_label(time_lag=time_lag, **kwargs)

        basename = "tmp_" + hashlib.md5(str(time.time()).encode()).hexdigest()
        self._basename.append(basename)
        seed = kwargs.get('seed', 42)

        spib_kernel(self._traj_data_list, self._traj_labels_list, time_lag,
                    base_path=basename, device=self._device,
                    **kwargs)

        prefix = f"{basename}/model_dt_{time_lag}"
        postfix = f"{seed}.npy"

        return SPIBResult(prefix, postfix, self._n_traj, dt=time_lag, 
                          b=self._b.cpu().numpy(), 
                          k=self._k.cpu().numpy())
    
    def __del__(self):
        for b in self._basename:
            shutil.rmtree(b, ignore_errors=True)