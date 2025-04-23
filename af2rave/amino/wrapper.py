'''
Wrapper module for AMINO.

This module defines a AMINO class that stores the model parameters.
The default parameters are best suited for af2rave use and may not be
universially applicable. For more general use, please use the cli module
by calling `af2rave amino`
'''

from __future__ import annotations

from ..colvar import Colvar
from . import amino
from af2rave.feature import utils

import numpy as np
import mdtraj as md

class AMINO(object):
    '''
    AMINO module reduces the redudancy in choice of collective variables.
    Usually the default parameters are best suited for af2rave use. 
    If you have memory issues, consider reducing the number of bins or striding the data.
    The performance of the code is O(N^2M),
    where N is the number of order parameters and M is the number of data points.
    The memory bottleneck is the mutual information calculation.

    :param n: The maximum number of order parameters to consider. Default is 20.
    :type n: int
    :param bins: The number of bins for the computing the mutual information. Default is 50.
    :type bins: int
    :param verbose: Whether to print the progress. Default is False.
    :type verbose: bool
    '''

    def __init__(self, **kwargs) -> None:

        self._n = kwargs.get('n', 20)
        self._bins = kwargs.get('bins', 50)
        self._verbose = kwargs.get('verbose', False)
        self._distance_matrix = kwargs.get('distance_matrix', None)
        self._names = kwargs.get('names', None)

        self._colvar = Colvar()
        self._result = None

    @property
    def result(self) -> list[str]:
        '''
        The resulting order parameters from AMINO.

        :return: The list of order parameters.
        :rtype: list[str]
        '''
        if self._result is None:
            raise ValueError("Please run AMINO first.")
        return self._result

    def run(self, label: list[str], data) -> None:
        '''
        Run AMINO on the given data. This is not the recommended way to use AMINO.
        Consider using the class methods `from_file` and `from_colvar` instead.

        :param label: The list of order parameter labels.
        :type label: list[str]
        :param data: The data of the order parameters, shaped (n_order_parameters, n_data_points)
        :type data: NDArray
        '''

        ops = [amino.OrderParameter(l, d) for l, d in zip(label, data)]
        result = amino.find_ops(ops, self._n, self._bins, verbose=self._verbose)
        self._result = [i.name for i in result]

    @classmethod
    def from_file(cls, filename: str | list[str], **kwargs) -> AMINO:
        '''
        Run AMINO from a COLVAR file. For keyword arguments, see `__init__`.

        :param filename: The COLVAR file or files to read.
        :type filename: str | list[str]
        :return: AMINO object, with the result stored in `result`.
        :rtype: AMINO
        '''

        colvar = Colvar()

        if isinstance(filename, str):
            colvar = Colvar.from_file(filename)
        else:
            for f in filename:
                colvar.tappend(Colvar.from_file(f))
        
        return AMINO.from_colvar(colvar, **kwargs)

    @classmethod
    def from_colvar(cls, colvar: Colvar, **kwargs) -> AMINO:
        '''
        Run AMINO from a Colvar object. For keyword arguments, see `__init__`.

        :param colvar: Colvar object.
        :type colvar: Colvar or str
        :return: AMINO object, with the result stored in `result`.
        :rtype: AMINO
        '''

        if not isinstance(colvar, Colvar):
            raise ValueError("The input should be a Colvar object.")

        instance = cls(**kwargs)

        instance._colvar = colvar
        instance.run(colvar.header, colvar.data)

        return instance
    
    @classmethod
    def from_dm(cls, 
                distance_matrix: str | np.array, 
                names: list[str] | str | Colvar,
                **kwargs) -> AMINO:
        
        '''
        Run AMINO from a distance matrix. For keyword arguments, see `__init__`.
        :param distance_matrix: The distance matrix to read.
        :type distance_matrix: str | np.ndarray
        :param names: The list of order parameter names. 
            If a Colvar object or filename is given, the header will be used.
        :type names: list[str] | str | Colvar
        :return: AMINO object, with the result stored in `result`.
        :rtype: AMINO
        '''
        
        if isinstance(names, str):
            _names = Colvar.from_file(names).header
        elif isinstance(names, list):
            _names = names
        elif isinstance(names, Colvar):
            _names = names.header
        else:
            raise ValueError("Unrecognized name.")

        if isinstance(distance_matrix, str):
            _dm = np.fromfile(distance_matrix)
        elif isinstance(distance_matrix, np.ndarray):
            _dm = distance_matrix
        else:
            raise ValueError("Unrecognized distance matrix.")
        
        # check dimensions, reshape if possible
        if len(_dm.shape) != 2:
            _dm = _dm.reshape(-1)
            length = int(np.sqrt(_dm.shape[0]))
            if length * length != _dm.shape[0]:
                raise ValueError("Distance matrix cannot be reshaped into a square matrix.")
            _dm = _dm.reshape(length, length)
        
        instance = cls(**kwargs)
        instance._result = amino.find_ops(distance_matrix=_dm,
                                     names=_names,
                                     max_outputs=instance._n,
                                     verbose=instance._verbose
                                     )
        
        return instance

    def to_colvar(self) -> Colvar:
        '''
        Output the chosen order parameters and their input values as a Colvar object.
        Equivalent to `input_colvar.choose(self.result)`.

        :return: The Colvar object with the chosen order parameters.
        :rtype: Colvar
        '''
        return self._colvar.choose(self.result)
    
    def explaination(self, topology: str | md.Topology) -> list[str]:
        '''
        Explain the order parameters in terms of the input topology.
        This is a helper function to understand the order parameters.
        
        :param topology: The topology file or object.
        :type topology: str | md.Topology
        :return: The list of order parameters.
        :rtype: list[str]
        '''

        if isinstance(topology, str):
            _top = md.load(topology).topology
        elif isinstance(topology, md.Topology):
            _top = topology
        else:
            raise ValueError("Unrecognized topology.")
        
        idx = [(int(i), int(j)) for _, i, j in [p.split('_') for p in self.result]]
        expl = ["distance {} {}".format(
            utils.chimera_representation(_top, i),
            utils.chimera_representation(_top, j)
        ) for i, j in idx]

        return expl

    def explain(self, topology: str | md.Topology) -> None:
        '''
        Print the explaination of the order parameters in terms of the input topology.
        
        :param topology: The topology file or object.
        :type topology: str | md.Topology
        '''

        for s in self.explaination(topology):
            print(s)