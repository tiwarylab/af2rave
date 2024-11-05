'''
This class handles a PLUMED style COLVAR file.
'''

import numpy as np


class Colvar(object):

    def __init__(self, header=[], time=np.array([]), data=np.array([])):
        self._header = header
        self._time = time
        self._data = data

    @classmethod
    def from_file(cls, filename: str):
        colvar = cls()
        colvar.read(filename)
        return colvar

    def _get_header_from_file(self):

        with open(self._filename, 'r') as file:
            headers = file.readline().strip().split()

        # check for redudant headers
        if len(headers) != len(set(headers)):
            raise ValueError("Non-unique metadate found in the file.")

        return headers[2:]

    def stride(self, interval: int):
        self._data = self._data[::interval]
        self._time = self._time[::interval]
        return self

    def read(self, filename: str, stride: int = 1):

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

    def write(self, filename: str, with_time: bool = True):
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

    def tappend(self, data: "Colvar") -> None:
        '''
        Append the data along the time axis in place.
        The incoming data should contain all columns in the base data.

        :param data: The incoming data.
        :type data: Colvar
        :return: None
        :raises ValueError: If the incoming data does not have all columns in the base data.
        '''

        index = self._match_header(self._header, data.header)
        if index is None:
            raise ValueError("The incoming data does not contain all the columns of the base data.")

        self._data = np.append(self._data, data._data[index], axis=1)
        self._time = np.append(self._time, data._time)

    def kappend(self, data: 'Colvar') -> None:

        if self.shape[1] != data.shape[1]:
            raise ValueError("The incoming date does not have the same number of entries as the base data.")

        self._header += data.header
        self._data = np.append(self._data, data._data, axis=0)

    def choose(self, columns: list) -> "Colvar":
        '''
        Choose the columns from the data. Returns a copy including the time.
        '''

        new_colvar = Colvar()
        new_colvar._time = self._time

        index = [self._header.index(i) for i in columns]
        new_colvar._header = columns
        new_colvar._data = self._data[index]

        return new_colvar

    def map(self, func: callable, insitu=True):
        '''
        Apply a function to the data. The function should take in a numpy array and return a numpy array.
        '''
        if insitu:
            self._data = func(self._data)
        else:
            return Colvar(self._header, self._time, func(self._data))

    @property
    def header(self):
        return self._header

    @property
    def shape(self):
        return self._data.shape

    @property
    def data(self):
        return self._data

    # python magic functions
    # --------------------------------
    def __contains__(self, key):
        return key in self._header

    def __getitem__(self, key):
        if key in self._header:
            return self._data[self._header.index(key)]
        else:
            raise KeyError(f"{key} does not exist.")

    def __setitem__(self, key, value):
        if len(value) != self.shape[1]:
            raise ValueError("The incoming data does not have the same number of entries as the base data.")
        if key not in self._header:
            self._header.append(key)
            self._data = np.append(self._data, value.reshape(1, -1), axis=0)
        else:
            self._data[self._header.index(key)] = value

    def __delitem__(self, key):
        if key in self._header:
            idx = self._header.index(key)
            self._header.pop(idx)
            self._data = np.delete(self._data, idx, axis=0)
        else:
            raise KeyError(f"{key} does not exist.")
