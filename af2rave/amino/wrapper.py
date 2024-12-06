'''
Wrapper module for AMINO.

This module defines a AMINO class that stores the model parameters.
The default parameters are best suited for af2rave use and may not be
universially applicable. For more general use, please use the cli module
by calling `af2rave amino`
'''

from __future__ import annotations

from ..colvar import Colvar
try:
    import amino
except ImportError:
    from . import amino

class AMINO(object):

    def __init__(self, **kwargs) -> None:
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
        :param kde_bandwidth: The bandwidth for the kernel density estimation. Default is 0.02.
        :type kde_bandwidth: float
        :param verbose: Whether to print the progress. Default is False.
        :type verbose: bool
        '''

        self._n = kwargs.get('n', 20)
        self._bins = kwargs.get('bins', 50)
        self._kde_bandwidth = kwargs.get('kde_bandwidth', 0.02)
        self._verbose = kwargs.get('verbose', False)

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
        result = amino.find_ops(
            ops, self._n, self._bins,
            bandwidth=self._kde_bandwidth, verbose=self._verbose
        )
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

        amino = cls(**kwargs)

        amino._colvar = colvar
        amino.run(colvar.header, colvar.data)

        return amino

    def to_colvar(self) -> Colvar:
        '''
        Output the chosen order parameters and their input values as a Colvar object.
        Equivalent to `input_colvar.choose(self.result)`.

        :return: The Colvar object with the chosen order parameters.
        :rtype: Colvar
        '''
        return self._colvar.choose(self.result)
