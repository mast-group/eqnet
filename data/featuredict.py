from collections import Counter

import numpy as np


class FeatureDictionary:
    """
    A simple feature dictionary that can convert features (ids) to
    their textual representation and vice-versa.
    """

    def __init__(self, add_unk=True):
        self.next_id = 0
        self.token_to_id = {}
        self.id_to_token = {}
        if add_unk:
            self.add_or_get_id(self.get_unk())

    def add_or_get_id(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]

        this_id = self.next_id
        self.next_id += 1
        self.token_to_id[token] = this_id
        self.id_to_token[this_id] = token

        return this_id

    def is_unk(self, token: str) -> bool:
        return token not in self.token_to_id

    def get_id_or_unk(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return self.token_to_id[self.get_unk()]

    def get_id_or_none(self, token: str):
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return None

    def get_name_for_id(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __str__(self):
        return str(self.token_to_id)

    def get_all_names(self) -> frozenset:
        return frozenset(self.token_to_id.keys())

    @staticmethod
    def get_unk() -> str:
        return "%UNK%"

    @staticmethod
    def get_feature_dictionary_for(tokens, count_threshold=5):
        token_counter = Counter(tokens)
        feature_dict = FeatureDictionary(add_unk=count_threshold > 0)
        for token, count in token_counter.items():
            if count >= count_threshold:
                feature_dict.add_or_get_id(token)
        return feature_dict


def get_empirical_distribution(element_dict: FeatureDictionary, elements, dirichlet_alpha=10.):
    """
    Retrieve empirical distribution of a seqence of elements
    :param element_dict: a dictionary that can convert the elements to their respective ids.
    :param elements: an iterable of all the elements
    :return:
    """
    targets = np.array([element_dict.get_id_or_unk(t) for t in elements])
    empirical_distribution = np.bincount(targets, minlength=len(element_dict)).astype(float)
    empirical_distribution += dirichlet_alpha / len(empirical_distribution)
    return empirical_distribution / (np.sum(empirical_distribution) + dirichlet_alpha)
