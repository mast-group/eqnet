import pickle

import numpy as np


class AbstractEncoder:
    def get_representation_vector_size(self) -> int:
        """Return the size of the representation"""
        raise NotImplementedError()

    def get_encoding(self, data: tuple) -> np.array:
        """
        Return the encoded representation
        :param data a tuple containing (list-of-tokens, Node) of the code snippet to be encoded
        :return a vector
        """
        raise NotImplementedError()

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
