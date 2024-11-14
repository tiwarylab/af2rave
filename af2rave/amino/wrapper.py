'''
Wrapper module for AMINO.

This module defines a AMINO class that stores the model parameters.
The default parameters are best suited for af2rave use and may not be
universially applicable. For more general use, please use the cli module
by calling `python -m af2rave.amino`
'''

from ..colvar import Colvar
try:
    import amino
except ImportError:
    from . import amino


class AMINO(object):

    def __init__(self, **kwargs) -> "AMINO":

        self._n = kwargs.get('n', 20)
        self._bins = kwargs.get('bins', 50)
        self._kde_bandwidth = kwargs.get('kde_bandwidth', 0.02)
        self._verbose = kwargs.get('verbose', False)

        self._colvar = Colvar()
        self._result = None

    @property
    def result(self) -> list[str]:
        if self._result is None:
            raise ValueError("Please run AMINO first.")
        return self._result

    def _run_from_self(self) -> None:
        ops = [amino.OrderParameter(ll, dd) for ll, dd in zip(self._label, self._data)]
        result = amino.find_ops(
            ops, self._n, self._bins,
            bandwidth=self._kde_bandwidth, verbose=self._verbose
        )
        self._result = [i.name for i in result]

    def from_timeseries(self, label, data) -> list[str]:

        if len(label) != len(data):
            raise ValueError("The length of label and data do not match.")
        self._label = label
        self._data = data
        self._colvar._data = self._data
        self._colvar.header = self.label
        self._run_from_self()

        return self.result

    def from_colvar(self, colvar: 'Colvar') -> list[str]:
        '''
        Run AMINO from a Colvar object or a COLVAR file.

        :param colvar: Colvar object or a COLVAR file.
        :type colvar: Colvar or str
        :return: List of AMINO order parameters.
        :rtype: list[str]
        '''

        if isinstance(colvar, Colvar):
            self._colvar = colvar
        else:
            self._colvar = Colvar(colvar)
        self._data = colvar._data
        self._label = colvar.header
        self._run_from_self()

        return self.result

    def get_colvar(self) -> Colvar:
        return self._colvar.choose(self.result)

    def write_colvar(self, filename: str) -> None:
        self.get_colvar().write(filename)
