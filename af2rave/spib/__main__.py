'''
TODO: reorganize this function and `SPIBProcess` to allow more fluent UI.
'''

import numpy as np
import torch
import os

import typer
from typer import Option, Argument
from typing import Annotated

from .wrapper import spib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")


def main(traj: Annotated[str, Argument(help="Path to the trajectory data. Can be a list of paths like traj1.dat,traj2.dat")],
         dt_list: Annotated[int, Argument(help="Time delay in terms of frames in the trajectory.")],
         label: Annotated[str, Option(help="Path to the initial state labels. Can be a list.")] = None,
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

    # parse the input arguments
    traj = traj.split(',')
    label = label.split(',')
    dt_list = [float(i) for i in dt_list.split(',')]

    # Load the data
    traj_data_list = [torch.from_numpy(np.load(f)).float().to(device) for f in traj]
    traj_labels_list = [torch.from_numpy(np.load(f)).float().to(device) for f in label]

    if bias is None:
        traj_weights_list = None
        base_path = os.path.join(base_path, "Unweighted")
    else:
        bias = eval(bias)
        if isinstance(bias, str):
            bias = [bias]
        traj_weights_list = [torch.from_numpy(np.load(file_path)).float().to(device) for file_path in bias]
        base_path = os.path.join(base_path, "Weighted")

    spib(traj_data_list, traj_labels_list, dt_list, batch_size, traj_weights_list,
         base_path, t0, RC_dim, encoder_type, neuron_num1, neuron_num2,
         threshold, patience, refinements, log_interval,
         lr_scheduler_step_size, lr_scheduler_gamma, learning_rate,
         beta, seed, UpdateLabel, SaveTrajResults)


if __name__ == '__main__':
    typer.run(main)
