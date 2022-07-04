import numpy as np
import pandas as pd
import math

from util import try_float

class NumpyGrouper:
    #http://esantorella.com/2016/06/16/groupby/
    def __init__(self, keys):
        self.unique_keys, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int) + 1
        self.set_indices()
        
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def __apply_no_return(self, function, vector):
        for idx in self.indices:
            function(vector[idx])
        return

    def __apply_to_dict(self, function, vector):
        result = {}
        for k, idx in enumerate(self.indices):
            result[self.unique_keys[k]] = function(vector[idx])
        return result

    def __apply_match_to_original(self, function, vector):
        result = np.zeros(len(self.keys_as_int))
        for idx in self.indices:
            grp_res = function(vector[idx])
            result[idx] = grp_res
        return result

    def apply(self, function, vector, return_val=True, match_to_original=True):

        if not isinstance(vector, (np.ndarray, np.generic)):
            vector = np.array(vector)

        if not return_val:
            return self.__apply_no_return(function, vector)

        if match_to_original:
            return self.__apply_match_to_original(function, vector)

        return self.__apply_to_dict(function, vector)


class Coord:
    def __init__(self, coords):
        self.x = try_float(coords[0])
        self.y = try_float(coords[1])

    def __eq__(self, other):
        if not isinstance(other, Coord):
            return False

        x_match = math.isclose(other.x, self.x)
        y_match = math.isclose(other.y, self.y)
        return x_match and y_match