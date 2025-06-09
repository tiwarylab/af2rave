'''
The SPIB package wrapper that deals with the low-level details of the SPIB model.
I/O is not taken care of here, so another application level wrapper is needed.
'''

import numpy as np
import torch
import os
import random

from .modules import SPIB
from .modules import SPIB_training


def spib(traj_data_list: list[torch.Tensor],
         traj_labels_list: list[torch.Tensor],
         dt: int,
         device: torch.device,
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
         SaveTrajResults: bool = True,
         **kwargs):

    '''
    Python interface for the SPIB model.
    '''

    # Check the dimensions of the input
    assert len(traj_data_list) == len(traj_labels_list)
    if traj_weights_list is not None:
        assert len(traj_weights_list) == len(traj_labels_list)

    output_dim = traj_labels_list[0].shape[1]

    os.makedirs(base_path, exist_ok=True)
    base_path = os.path.join(base_path, "model")

    final_result_path = base_path + '_result.dat'
    with open(final_result_path, 'w') as f:
        f.write("Final Result\n")

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    data_init_list = []
    if traj_weights_list is None:
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

    output_path = base_path + f"_dt_{dt}"

    IB = SPIB.SPIB(encoder_type, RC_dim, output_dim, data_shape, device,
                    UpdateLabel, neuron_num1, neuron_num2)

    IB.to(device)

    # use the training set to initialize the pseudo-inputs
    IB.init_representative_inputs(train_past_data, train_data_labels)

    train_result = False
    train_result = SPIB_training.train(IB, beta,
                                        train_past_data, train_future_data, train_data_labels, train_data_weights,
                                        test_past_data, test_future_data, test_data_labels, test_data_weights,
                                        learning_rate, lr_scheduler_step_size, lr_scheduler_gamma,
                                        batch_size, threshold, patience, refinements,
                                        output_path, log_interval, device, seed)

    if train_result:
        return

    SPIB_training.output_final_result(IB, device,
                                        train_past_data, train_future_data, train_data_labels, train_data_weights,
                                        test_past_data, test_future_data, test_data_labels, test_data_weights,
                                        batch_size, output_path, final_result_path, dt, beta, learning_rate, seed)

    for i in range(len(traj_data_list)):
        IB.save_traj_results(traj_data_list[i], batch_size, output_path, SaveTrajResults, i, seed)

    IB.save_representative_parameters(output_path, seed)
