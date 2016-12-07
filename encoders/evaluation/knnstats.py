from collections import defaultdict, OrderedDict

import numpy as np
from scipy.spatial.distance import cdist, squareform, pdist

from data.dataimport import import_data
from encoders.baseencoder import AbstractEncoder


class SemanticEquivalentDistanceEvaluation:
    """
    Compute the ratio of the distance between semantically equivalent code and non-semantically equivalent code.
    This evaluation assumes that the noise samples are semantically equivalent.
    """

    def __init__(self, encoder_filename: str, encoder: AbstractEncoder = None):
        if encoder is None:
            self.__encoder = AbstractEncoder.load(encoder_filename)
        else:
            self.__encoder = encoder

    def evaluate(self, data_filename: str, consider_only_first_n_components: int = None, num_nns: int = 10) -> np.array:
        data = import_data(data_filename)
        encodings = []
        equivalent_to = []
        equivalence_sets = []
        for name, code in data.items():
            idx = len(encodings)
            enc = self.__encoder.get_encoding(code['original'])
            assert not np.isnan(np.sum(enc))
            encodings.append(enc)
            equivalent_to.append(idx)
            for noisy_sample in code['noise']:
                enc = self.__encoder.get_encoding(noisy_sample)
                assert not np.isnan(np.sum(enc))
                encodings.append(enc)
                equivalent_to.append(idx)

            equivalence_sets.append(set(range(idx, len(encodings))))

        encodings = np.array(encodings)
        if consider_only_first_n_components is not None:
            encodings = encodings[:, :consider_only_first_n_components]

        all_distances = squareform(pdist(encodings, 'cosine'))  # TODO: avoid square form somehow
        assert not np.any(np.isnan(all_distances))
        identity = np.arange(all_distances.shape[0])
        all_distances[identity, identity] = float(
            'inf')  # The distance to self is infinite to get the real neighbors in the next step
        k_nearest_neighbor_idxs = np.argpartition(all_distances, num_nns)[:, :num_nns]

        left_index = np.atleast_2d(identity).T
        order_of_knearest_neighbors = np.argsort(all_distances[left_index, k_nearest_neighbor_idxs])
        all_distances = k_nearest_neighbor_idxs[left_index, order_of_knearest_neighbors]

        equivalent_elements = {}
        for eq_set in equivalence_sets:
            for element in eq_set:
                equivalent_elements[element] = eq_set

        k_nns_semantic_eq = np.zeros(num_nns, dtype=np.float64)
        num_k_nns = np.zeros(num_nns)
        for i in range(all_distances.shape[0]):
            semantically_eq_nns = equivalent_elements[i]
            if len(semantically_eq_nns) < 2:
                continue
            for j in range(num_nns):
                num_k_nns[j] += 1
                k_nns_semantic_eq[j] += float(len(semantically_eq_nns & set(all_distances[i, :j + 1]))) / \
                                        min(len(semantically_eq_nns), j + 1)

        return k_nns_semantic_eq / num_k_nns

    def evaluate_with_test(self, data_filename: str, test_filename: str, consider_only_first_n_components: int = None,
                           num_nns: int = 15) -> np.array:
        test_data = import_data(test_filename)
        test_samples = defaultdict(set)  # eq_class -> tokens
        for eq_class, code in test_data.items():
            test_samples[eq_class].add(''.join(code['original'][0]))
            for sample in code['noise']:
                test_samples[eq_class].add(''.join(sample[0]))

        data = import_data(data_filename)
        encodings = []
        equivalence_classes = defaultdict(set)  # eq_class->set(ids)
        test_samples_idx_map = OrderedDict()  # id-> eq_class
        for eq_class, code in data.items():
            encoding = self.__encoder.get_encoding(code['original'])
            assert not np.isnan(np.sum(encoding))
            encodings.append(encoding)
            equivalence_classes[eq_class].add(len(encodings) - 1)
            if ''.join(code['original'][0]) in test_samples[eq_class]:
                test_samples_idx_map[len(encodings) - 1] = eq_class
            for noisy_sample in code['noise']:
                encoding = self.__encoder.get_encoding(noisy_sample)
                assert not np.isnan(np.sum(encoding))
                encodings.append(encoding)
                equivalence_classes[eq_class].add(len(encodings) - 1)
                if ''.join(noisy_sample[0]) in test_samples[eq_class]:
                    test_samples_idx_map[len(encodings) - 1] = eq_class

        test_sample_idxs = np.fromiter(test_samples_idx_map.keys(), dtype=np.int32)
        encodings = np.array(encodings)
        if consider_only_first_n_components is not None:
            encodings = encodings[:, :consider_only_first_n_components]

        nearest_neighbors = cdist(encodings[test_sample_idxs], encodings, 'cosine')  # TODO: avoid square form somehow
        identity = np.arange(nearest_neighbors.shape[0])
        assert nearest_neighbors.shape[0] == len(test_sample_idxs)
        nearest_neighbors[identity, test_sample_idxs] = float(
            'inf')  # The distance to self is infinite to get the real neighbors in the next step

        k_nearest_neighbor_idxs = np.argpartition(nearest_neighbors, num_nns)[:, :num_nns]

        left_index = np.atleast_2d(identity).T
        order_of_knearest_neighbors = np.argsort(nearest_neighbors[left_index, k_nearest_neighbor_idxs])
        nearest_neighbors = k_nearest_neighbor_idxs[left_index, order_of_knearest_neighbors]

        k_nns_semantic_eq = np.zeros(num_nns, dtype=np.float64)
        num_k_nns = np.zeros(num_nns)
        for i in range(nearest_neighbors.shape[0]):
            test_sample_i = test_sample_idxs[i]
            semantically_eq_nns = equivalence_classes[test_samples_idx_map[test_sample_i]]
            if len(semantically_eq_nns) < 2:
                continue
            for j in range(num_nns):
                num_k_nns[j] += 1
                k_nns_semantic_eq[j] += float(len(semantically_eq_nns & set(nearest_neighbors[i, :j + 1]))) / \
                                        min(len(semantically_eq_nns), j + 1)

        return k_nns_semantic_eq / num_k_nns


if __name__ == "__main__":
    import sys

    if 6 > len(sys.argv) < 4:
        print("Usage <encoderPkl> <evaluationFilename> <allFilename> [considerOnlyFirstKcomponents]")
        sys.exit(-1)

    evaluator = SemanticEquivalentDistanceEvaluation(sys.argv[1])
    if len(sys.argv) == 5:
        n_components = int(sys.argv[4])
        nn_stats = evaluator.evaluate_with_test(sys.argv[3], sys.argv[2], consider_only_first_n_components=n_components)
    else:
        nn_stats = evaluator.evaluate_with_test(sys.argv[3], sys.argv[2])
    print("Avg Semantically Equivalent NNs: %s" % nn_stats)
