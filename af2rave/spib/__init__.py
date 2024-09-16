"""
SPIB: A deep learning-based framework to learn RCs from MD trajectories. Code maintained by Dedi.

This updated wrapper is configured by Da Teng for future integration with AF2RAVE.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import numpy as np
import torch
import os
import random

import typer
from typer import Option, Argument
from typing import Annotated

from . import SPIB
from . import SPIB_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")

def main(traj: Annotated[str, Argument(help="Path to the trajectory data. Can be a list of paths like traj1.npy,traj2.npy")],
         label: Annotated[str, Argument(help="Path to the initial state labels. Can be a list.")],
         dt_list: Annotated[str, Argument(help="Time delay in terms of frames in the trajectory. Can be a list like 10,20,30")],
         batch_size: Annotated[int, Option("--bs", help="Training batch size.")] = 512,
         bias: Annotated[str, Option(help="Path to the weights of the samples. Leave blank is simulation is unbiased.")] = None,
         base_path: Annotated[str, Option(help="Path to the output base directory")] = "SPIB",
         t0: Annotated[int, Option(help="Start time in terms of frames in trajectory. The trajectory before frame t0 is ignored")] = 0,
         RC_dim: Annotated[int, Option("--d", help="Dimension of RC or bottleneck.")] = 2,
         encoder_type: Annotated[str, Option(help="Type of encoder ('Linear' or 'Nonlinear').")] = "Linear",
         neuron_num1: Annotated[int, Option("--n1", help="Number of nodes in each hidden layer of the encoder.")] = 32,
         neuron_num2: Annotated[int, Option("--n2", help="Number of nodes in each hidden layer of the encoder.")] = 128,
         threshold: Annotated[float, Option(help="Threshold in terms of the change of the predicted state population for measuring the convergence of the training")] = 0.01,
         patience: Annotated[int, Option(help="Number of epochs with the change of the state population smaller than the threshold after which this iteration of the training finishes")] = 2,
         refinements: Annotated[int, Option(help="Minimum refinements")] = 8,
         log_interval: Annotated[int, Option(help="Interval to save the model.")] = 10000,
         lr_scheduler_step_size: Annotated[int, Option(help="Period of learning rate decay")] = 5,
         lr_scheduler_gamma: Annotated[float, Option(help="Multiplicative factor of learning rate decay.")] = 0.8,
         learning_rate: Annotated[float, Option(help="Initial learning rate of Adam optimizer")] = 1e-4,
         beta: Annotated[float, Option("--b", help="Hyper-parameter beta")] = 0.05,
         seed: Annotated[int, Option(help="Random seed.")] = 42,
         UpdateLabel: Annotated[bool, Option("--update-label", help="Whether update the labels during the training process")] = True,
         SaveTrajResults: Annotated[bool, Option("--save-traj-results", help="Whether save trajectory results")] = True):
    
    # By default, we save all the results in subdirectories of the following path.
    base_path = "SPIB"

    # parse the input arguments
    traj = traj.split(',')
    label = label.split(',')
    dt_list = [int(i) for i in dt_list.split(',')]

    # Load the data
    traj_data_list = [torch.from_numpy(np.load(f)).float().to(device) for f in traj]
    traj_labels_list = [torch.from_numpy(np.load(f)).float().to(device) for f in label]
    assert len(traj_data_list) == len(traj_labels_list)
    
    if bias is None:
        traj_weights_list = None
        base_path = os.path.join(base_path, "Unweighted")
    else:
        bias = eval(bias)
        if isinstance(bias, str):
            bias = [bias]
        traj_weights_list = [torch.from_numpy(np.load(file_path)).float().to(device) for file_path in bias]
        base_path = os.path.join(base_path, "Weighted")

    assert len(traj_weights_list) == len(traj_labels_list)

    spib(traj_data_list, traj_labels_list, dt_list, batch_size, traj_weights_list, 
         base_path, t0, RC_dim, encoder_type, neuron_num1, neuron_num2, 
         threshold, patience, refinements, log_interval, 
         lr_scheduler_step_size, lr_scheduler_gamma, learning_rate, 
         beta, seed, UpdateLabel, SaveTrajResults)


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
    SPIB: A deep learning-based framework to learn RCs from MD trajectories. This is the Python interface for the SPIB model.
    '''

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


if __name__ == '__main__':
    typer.run(main)
