'''
Wrapper module for AMINO.

This module defines a AMINO class that stores the model parameters.
The default parameters are best suited for af2rave use and may not be
universially applicable. For more general use, please use the cli module
by calling `python -m af2rave.amino`
'''

from __future__ import annotations

from ..colvar import Colvar
try:
    import amino
except ImportError:
    from . import amino

from pathlib import Path

class AMINO(object):

    def __init__(self, **kwargs) -> None:

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

    def run(self, label, data) -> None:
        ops = [amino.OrderParameter(l, d) for l, d in zip(label, data)]
        result = amino.find_ops(
            ops, self._n, self._bins,
            bandwidth=self._kde_bandwidth, verbose=self._verbose
        )
        self._result = [i.name for i in result]

    @classmethod
    def from_file(cls, filename: str | list[str], **kwargs) -> AMINO:
        '''
        Run AMINO from a COLVAR file.

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
        Run AMINO from a Colvar object.

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
        return self._colvar.choose(self.result)
