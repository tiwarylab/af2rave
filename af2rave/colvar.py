'''
This class handles a PLUMED style COLVAR file.
'''

from __future__ import annotations

import numpy as np
from numpy._typing._array_like import NDArray


class Colvar(object):

    def __init__(self, header=[], time=np.array([]), data=np.array([])) -> None:
        self._header = header
        self._time = time.reshape(1, -1)
        self._data = data

    @classmethod
    def from_file(cls, filename: str) -> Colvar:
        colvar = cls()
        colvar.read(filename)
        return colvar

    def _get_header_from_file(self) -> list[str]:

        with open(self._filename, 'r') as file:
            headers = file.readline().strip().split()

        # check for redudant headers
        if len(headers) != len(set(headers)):
            raise ValueError("Non-unique metadate found in the file.")

        return headers[2:]

    def stride(self, interval: int) -> Colvar:
        self._data = self._data[::interval]
        self._time = self._time[::interval]
        return self

    def read(self, filename: str, stride: int = 1) -> None:

        self._filename = filename
        self._header = self._get_header_from_file()

        self._data = np.loadtxt(self._filename, unpack=True)
        if self.shape[0] != len(self._header):
            raise ValueError("Number of columns in the file does not match the number of headers."
                             f"Got {self.shape[0]} columns and {len(self._header)} headers.")

        if "time" in self._header:
            idx = self._header.index("time")
            self._time = self._data[idx]
            self._data = np.delete(self._data, idx, axis=0)
            self._header.pop(idx)

        # stride the data
        if stride > 1:
            self.stride(stride)

    def write(self, filename: str, with_time: bool = True) -> None:
        with open(filename, "w") as f:
            if with_time and self._time is not None:
                f.write("#! FIELDS time ")
                f.write(" ".join(self._header))
                f.write("\n")
                np.savetxt(f, np.vstack((self._time, self._data)).T, fmt="%.6f")
            else:
                f.write("#! FIELDS ")
                f.write(" ".join(self._header))
                f.write("\n")
                np.savetxt(f, self._data.T, fmt="%.6f")

    # several helper functions to help with appending operations
    # ---------------------------------------------------------
    @staticmethod
    def _match_header(base: list, new: list):
        '''
        This finds the index of the incoming columns in the base columns.
        The dimension of the incoming columns should be greater than or equal to the base columns.

        Returns an array with the size of the base columns, or None if the merge is impossible.
        The array contains the index of that column in the incoming columns.
        '''

        if len(base) > len(new):
            return None

        arg_arr = np.zeros(len(base), dtype=int)
        for i, k in enumerate(base):
            if k in new:
                arg_arr[new.index(k)] = i
            else:
                return None
        return arg_arr

    def tappend(self, data: Colvar | str, stride: int = None) -> Colvar:
        '''
        Append the data along the time axis in place.
        The incoming data should contain all columns in the base data.
        If the Colvar to append to is empty, the header will be copied from the incoming data.

        :param data: The incoming data. Either a colvar object or a filename.
        :type data: Colvar
        :param stride: The stride to apply to the incoming data.
        :type stride: int
        :return: Self
        :raises ValueError: If the incoming data does not have all columns in the base data.
        '''

        if isinstance(data, str):
            data = Colvar.from_file(data)

        if stride is not None:
            data.stride(stride)

        # if myself is initialized as empty, simply copy
        if len(self._header) == 0:
            self._header = data.header
            self._data = data.data
            self._time = data.time
            return self

        index = self._match_header(self._header, data.header)
        if index is None:
            raise ValueError("The incoming data does not contain all the columns of the base data.")

        self._data = np.append(self._data, data._data[index], axis=1)
        self._time = np.append(self._time, data._time)

        return self

    def kappend(self, data: Colvar | str) -> Colvar:
        '''
        Append the data along the column axis in place.
        The incoming data should have the same number of rows as the base data.

        :param data: The incoming data. Either a colvar object or a filename.
        :type data: Colvar | str
        :return: Self
        '''

        if isinstance(data, str):
            data = Colvar.from_file(data)

        if self.shape[1] != data.shape[1]:
            raise ValueError("The incoming date does not have the same number of entries as the base data.")

        self._header += data.header
        self._data = np.append(self._data, data._data, axis=0)

        return self

    def choose(self, columns: list) -> Colvar:
        '''
        Choose the columns from the data. Returns a copy including the time.
        '''

        new_colvar = Colvar()
        new_colvar._time = self._time

        index = [self._header.index(i) for i in columns]
        new_colvar._header = columns
        new_colvar._data = self._data[index]

        return new_colvar

    def map(self, func: callable, insitu=True) -> Colvar:
        '''
        Apply a function to the data. The function should take in a numpy array and return a numpy array.
        '''
        if insitu:
            self._data = func(self._data)
            return self
        else:
            return Colvar(self._header, self._time, func(self._data))

    @property
    def header(self) -> list[str]:
        return self._header

    @property
    def shape(self):
        return self._data.shape

    @property
    def data(self):
        return self._data
    
    @property
    def time(self):
        return self._time

    # python magic functions
    # --------------------------------
    def __contains__(self, key) -> bool:
        return key in self._header

    def __getitem__(self, key):
        if key in self._header:
            return self._data[self._header.index(key)]
        else:
            raise KeyError(f"{key} does not exist.")

    def __setitem__(self, key, value) -> None:
        if len(value) != self.shape[1]:
            raise ValueError("The incoming data does not have the same number of entries as the base data.")
        if key not in self._header:
            self._header.append(key)
            self._data = np.append(self._data, value.reshape(1, -1), axis=0)
        else:
            self._data[self._header.index(key)] = value

    def __delitem__(self, key) -> None:
        if key in self._header:
            idx = self._header.index(key)
            self._header.pop(idx)
            self._data = np.delete(self._data, idx, axis=0)
        else:
            raise KeyError(f"{key} does not exist.")
