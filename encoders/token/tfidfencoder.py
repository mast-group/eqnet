import pickle
from collections import Counter
from itertools import chain

import numpy as np

from data.dataimport import import_data
from data.featuredict import FeatureDictionary
from encoders.baseencoder import AbstractEncoder


class TfidfEncoder(AbstractEncoder):
    def decoder_loss(self, data: tuple, representation: np.array) -> float:
        raise Exception("TfidfEncoder has no decoder loss")

    def get_representation_vector_size(self) -> int:
        return len(self.__feature_dict)

    def get_encoding(self, data: tuple) -> np.array:
        document_tokens = Counter(self.__feature_dict.get_id_or_unk(t) for t in data[0])
        vect = np.zeros(len(self.__feature_dict), dtype=np.float)
        for word_id, count in document_tokens.items():
            vect[word_id] = count * self.__idfs[word_id]
        vect /= len(data[0])
        return vect

    def __init__(self, train_file):
        data = import_data(train_file)

        def document_tokens():
            for snippet in data.values():
                yield snippet['original'][0]

        all_document_tokens = [s for s in document_tokens()]
        self.__feature_dict = FeatureDictionary.get_feature_dictionary_for(chain(*all_document_tokens),
                                                                           count_threshold=10)

        self.__idfs = np.ones(len(self.__feature_dict), dtype=np.int)  # use 1s for smoothing
        for document in all_document_tokens:
            document_word_ids = set(self.__feature_dict.get_id_or_unk(t) for t in document)
            for word_id in document_word_ids:
                self.__idfs[word_id] += 1

        self.__idfs = np.log(self.__idfs.astype(np.float))

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
