'''
This class handles a PLUMED style COLVAR file.
'''

import numpy as np

class Colvar():

    def __init__(self, filename = None):
        self._filename = None
        self._header = []
        self._data = np.array([])
        
        if filename is not None:
            self.read(filename)

    def _get_header_from_file(self):

        with open(self._filename, 'r') as file:
            headers = file.readline().strip().split()

        # check for redudant headers
        if len(headers) != len(set(headers)):
            raise ValueError("Non-unique metadate found in the file.")

        return headers[2:]
    
    def stride(self, interval: int):
        self._data = self._data[::interval]
        return self

    def read(self, filename: str, stride: int= 1):

        self._filename = filename
        self._header = self._get_header_from_file()

        self._data = np.loadtxt(self._filename, unpack=True)[::stride]
        if self.shape[0] != len(self._header):
            raise ValueError("Number of columns in the file does not match the number of headers."
                             f"Got {self.shape[0]} columns and {len(self._header)} headers.")

    def write(self, filename):
        with open(filename, "w") as f:
            f.write("#! FIELDS ")
            f.write(" ".join(self._header))
            f.write("\n")
            np.savetxt(f, self._data, fmt="%.6f")

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
                arg_arr[base.index(i)] = i
            else:
                return None
        return arg_arr


    def tappend(self, data: 'Colvar'):
        
        index = self._match_header(self._header, data.header)
        if index is None:
            raise ValueError("The incoming data does not contain all the columns of the base data.")
        
        self._data = np.append(self._data, data._data[index], axis=1)
        return self

    def kappend(self, data: 'Colvar'):

        if self.shape[1] != data.shape[1]:
            raise ValueError("The incoming date does not have the same number of entries as the base data.")
        
        self._header += data.header
        self._data = np.append(self._data, data._data, axis=0)
        
    def choose(self, columns: list):
        '''
        Choose the columns from the data. Returns a copy including the time.
        '''
        index = [self._header.index(i) for i in columns]
        self._header = columns
        self._data = self._data[:, index]
    
    @property
    def header(self):
        return self._header
    
    @property
    def shape(self):
        return self._data.shape

    # python magic functions
    # --------------------------------
    def __contains__(self, key):
        return key in self._header

    def __getitem__(self, key):
        if key in self._header:
            return self._data[:, self._header.index(key)]
        else:
            raise KeyError(f"{key} does not exist.")
    
    def __setitem__(self, key, value):
        if key not in self._header:
            self._header.append(key)
            self._data = np.append(self._data, np.zeros((self._data.shape[0], 1)), axis=0)
        else:
            self._data[:, self._header.index(key)] = value
    
    def __delitem__(self, key):
        if key in self._header:
            idx = self._header.index(key)
            self._header.pop(idx)
            self._data = np.delete(self._data, idx, axis=0)
        else:
            raise KeyError(f"{key} does not exist.")