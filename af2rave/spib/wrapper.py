import numpy as np
import torch
import os
import random

from . import SPIB
from . import SPIB_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")

def spib(traj_data_list: list[torch.Tensor], 
         traj_labels_list: list[torch.Tensor], 
         dt_list = list[int], 
         batch_size: int = 512,
         traj_weights_list: list[torch.Tensor] = None,
         base_path: str = "SPIB",
         t0: int = 0,
         RC_dim: int = 2,
         encoder_type: str = "Linear",
         neuron_num1: int = 32,
         neuron_num2: int = 128,
         threshold: float = 0.01,
         patience: int = 2,
         refinements: int = 8,
         log_interval: int = 10000,
         lr_scheduler_step_size: int = 5,
         lr_scheduler_gamma: float = 0.8,
         learning_rate: float = 1e-4,
         beta: float = 0.05,
         seed: int = 42,
         UpdateLabel: bool = True,
         SaveTrajResults: bool = True):
    
    ''' 
    Python interface for the SPIB model.
    '''

    # Check the dimensions of the input
    assert len(traj_data_list) == len(traj_labels_list)
    if traj_weights_list is not None:
        assert len(traj_weights_list) == len(traj_labels_list)

    output_dim = traj_labels_list[0].shape[1]

    final_result_path = base_path + '_result.dat'
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
    with open(final_result_path, 'w') as f:
        f.write("Final Result\n")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    for dt in dt_list:
        data_init_list = [] 
        if traj_weights_list == None:
            data_init_list = [SPIB_training.data_init(t0, dt, trj, lbl, None) for trj, lbl in zip(traj_data_list, traj_labels_list)]
            train_data_weights = None
            test_data_weights = None
        else:
            data_init_list = [SPIB_training.data_init(t0, dt, trj, lbl, wts) for trj, lbl, wts in zip(traj_data_list, traj_labels_list, traj_weights_list)]
            train_data_weights = torch.cat([dat[4] for dat in data_init_list], dim=0)
            test_data_weights = torch.cat([dat[8] for dat in data_init_list], dim=0)

        data_shape = data_init_list[0][0]
        train_past_data = torch.cat([dat[1] for dat in data_init_list], dim=0)
        train_future_data = torch.cat([dat[2] for dat in data_init_list], dim=0)
        train_data_labels = torch.cat([dat[3] for dat in data_init_list], dim=0)

        test_past_data = torch.cat([dat[5] for dat in data_init_list], dim=0)
        test_future_data = torch.cat([dat[6] for dat in data_init_list], dim=0)
        test_data_labels = torch.cat([dat[7] for dat in data_init_list], dim=0)

        output_path = base_path + f"_d={RC_dim}_t={dt}_b={beta:.4f}_learn={learning_rate}"

        IB = SPIB.SPIB(encoder_type, RC_dim, output_dim, data_shape, device, \
                       UpdateLabel, neuron_num1, neuron_num2)
        
        IB.to(device)
        
        # use the training set to initialize the pseudo-inputs
        IB.init_representative_inputs(train_past_data, train_data_labels)

        train_result = False
        train_result = SPIB_training.train(IB, beta, 
                                            train_past_data, train_future_data, train_data_labels, train_data_weights, \
                                            test_past_data, test_future_data, test_data_labels, test_data_weights, \
                                            learning_rate, lr_scheduler_step_size, lr_scheduler_gamma,\
                                            batch_size, threshold, patience, refinements, \
                                            output_path, log_interval, device, seed)
        
        if train_result:
            return
        
        SPIB_training.output_final_result(IB, device, 
                                            train_past_data, train_future_data, train_data_labels, train_data_weights, \
                                            test_past_data, test_future_data, test_data_labels, test_data_weights, \
                                            batch_size, output_path, final_result_path, dt, beta, learning_rate, seed)

        for i in range(len(traj_data_list)):
            IB.save_traj_results(traj_data_list[i], batch_size, output_path, SaveTrajResults, i, seed)
        
        IB.save_representative_parameters(output_path, seed)

def create_input_from_colvar(filename: list[str] | str,
                             stride: int = 1) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    '''
    Create input data from one or a list of PLUMED-style COLVAR file.

    :param filename: The name of the file to read the CVs from.
    :type filename: list[str] | str
    :param stride: The interval at which to read the CVs. Default: 1
    :type stride: int
    :return: A tuple of two lists of torch.Tensor, the first list contains the CVs and the second list contains the labels.
    '''

    if isinstance(filename, str):
        filename = [filename]

    traj_data_list = []
    traj_labels_list = []

    n_states = len(filename) * 2

    for i, f in enumerate(filename):
        
        data = np.loadtxt(f)[::stride]

        n_data = data.shape[0]
        scalar_label = np.rint(np.linspace(0, 1, n_data)) + i * 2
        onehot_label = np.eye(n_states)[scalar_label.astype(int)]

        data = torch.tensor(data, dtype=torch.float32).to(device)
        label = torch.tensor(onehot_label, dtype=torch.float32).to(device)

        traj_data_list.append(data)
        traj_labels_list.append(label)
    
    return traj_data_list, traj_labels_list
