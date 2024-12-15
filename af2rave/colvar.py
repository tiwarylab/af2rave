'''
This class handles a PLUMED style COLVAR file.
'''

from __future__ import annotations

import numpy as np
from numpy._typing._array_like import NDArray
from pathlib import Path


class Colvar(object):

    def __init__(self, header=[], time=None, data=np.array([])) -> None:
        self._header = header
        if time is not None:
            self._time = time.reshape(1, -1)
        else:
            self._time = None
        self._data = data

    @classmethod
    def from_file(cls, filename: str) -> Colvar:
        '''
        Construct a Colvar object from a file. Same as Colvar().read(filename).

        :param filename: The filename to read.
        :type filename: str
        '''
        colvar = cls()
        colvar.read(filename)
        return colvar

    def _get_header_from_file(self) -> list[str]:

        with open(self._filename, 'r') as file:
            headers = file.readline().strip().split()

        # check for redudant headers
        if len(headers) != len(set(headers)):
            raise ValueError("Non-unique metadata found in the file.")

        return headers[2:]

    def stride(self, interval: int) -> Colvar:
        '''
        In-place stride the data with a given interval.

        :param interval: The interval to stride the data.
        :type interval: int
        :return: Self
        '''
        self._data = self._data[:, ::interval]
        if self._time is not None:
            self._time = self._time[::interval]
        return self

    def read(self, filename: str, stride: int = 1) -> Colvar:
        '''
        Read a PLUMED-style colvar file into the object.

        :param filename: The filename to read.
        :type filename: str
        :param stride: The stride to apply to the data. (Optional, default=1)
        :type stride: int
        :return: Self
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the number of columns in the file does not match the number of headers.
        '''

        self._filename = filename
        if not Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist.")

        self._header = self._get_header_from_file()

        self._data = np.loadtxt(self._filename, unpack=True).reshape(len(self._header), -1)
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
        
        return self

    def write(self, filename: str, with_time: bool = True) -> None:
        '''
        Write the Colvar object to a file. This will overwrite the file if it exists.

        :param filename: The filename to write to.
        :type filename: str
        :param with_time: If True, write the time data as well.
        :type with_time: bool
        '''
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
    
    def _get_index(self, key: str) -> int:
        '''
        Get the index of the key in the header.
        This function handles exceptions where the key does not exist.

        :param key: The key to search for.
        :type key: str
        :return: The index of the key.
        :rtype: int
        :raises KeyError: If the key does not exist.
        '''
        try:
            return self._header.index(key)
        except ValueError:
            raise KeyError(f"{key} does not exist.")

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

        # check the time, either both have time or none
        if (self._time is None) != (data.time is None):
            raise ValueError("Both data should have time or none.")

        self._data = np.append(self._data, data._data[index], axis=1)
        if self._time is not None:
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

        index = [self._get_index(i) for i in columns]
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
        '''
        The header of the Colvar. Note that time is not considered one of the fields.

        :return: The header of the Colvar.
        :rtype: list[str]
        '''
        return self._header

    @property
    def shape(self) -> tuple[int]:
        '''
        The shape of the header. The first dimension is the number of fields, the second dimension is the number of rows.

        :return: The shape of the data.
        :rtype: tuple[int]
        '''
        return self._data.shape

    @property
    def data(self):
        '''
        Return the data of the Colvar.
        '''
        return self._data
    
    @property
    def time(self):
        '''
        Return the time data of the Colvar. Can be None if the data does not have time.

        :return: The time data.
        :rtype: numpy.ndarray | None
        '''
        return self._time

    # python magic functions
    # --------------------------------
    def __contains__(self, key) -> bool:
        return key in self._header

    def __getitem__(self, key):
        return self._data[self._get_index(key)]

    def __setitem__(self, key, value) -> None:
        # first check if we are empty
        if len(self._header) == 0:
            self._header = [key]
            self._data = value.reshape(1, -1)
            return
        if len(value) != self.shape[1]:
            raise ValueError("The incoming data does not have the same number of entries as the base data.")
        if key not in self._header:
            self._header.append(key)
            self._data = np.append(self._data, value.reshape(1, -1), axis=0)
        else:
            self._data[self._header.index(key)] = value

    def __delitem__(self, key) -> None:
        idx = self._get_index(key)
        self._header.pop(idx)
        self._data = np.delete(self._data, idx, axis=0)
