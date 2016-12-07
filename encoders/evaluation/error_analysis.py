import sys
from collections import defaultdict
from itertools import product

import numpy as np
from scipy.spatial.distance import cdist

from data.dataimport import import_data
from encoders.baseencoder import AbstractEncoder


def get_dataset_samples(filename: str):
    dataset_samples = []
    for name, code in import_data(filename).items():
        dataset_samples.append(''.join(code['original'][0]))
        for noisy_sample in code['noise']:
            dataset_samples.append(''.join(noisy_sample[0]))
    return set(dataset_samples)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage <encoderPkl> <dataset.json.gz> <validationset.json.gz> <testset.json.gz> <neweqtestset.json.gz>")
        sys.exit(-1)

    num_nns = 1
    validation_samples = get_dataset_samples(sys.argv[3])
    testset_samples = get_dataset_samples(sys.argv[4])
    newewset_samples = get_dataset_samples(sys.argv[5])

    data = import_data(sys.argv[2])
    encoder = AbstractEncoder.load(sys.argv[1])

    expression_data, encodings = [], []
    eq_class_idx_to_names = {}
    eq_class_counts = defaultdict(int)


    def add_sample(data, eq_class_idx: int):
        expression = ''.join(data[0])

        sample_data = dict(
            expr=expression,
            is_val=expression in validation_samples,
            is_test=expression in testset_samples,
            is_neweq=expression in newewset_samples,
            eq_class=eq_class_idx
        )
        expression_data.append(sample_data)

        representation = encoder.get_encoding(data)
        assert not np.isnan(np.sum(representation))
        encodings.append(representation)


    for eq_class_idx, (name, code) in enumerate(data.items()):
        eq_class_idx_to_names[eq_class_idx] = name
        eq_class_counts[eq_class_idx] = 1 + len(code['noise'])
        add_sample(code['original'], eq_class_idx)
        for noisy_sample in code['noise']:
            add_sample(noisy_sample, eq_class_idx)

    encodings = np.array(encodings)

    knns = np.zeros((encodings.shape[0], num_nns), dtype=np.int)
    knn_distances = np.zeros((encodings.shape[0], num_nns))
    for i in range(encodings.shape[0]):
        distances = cdist(encodings[[i]], encodings)
        distances[0, i] = float('inf')
        distances = distances[0]
        knns[i] = np.argsort(distances)[:num_nns]
        knn_distances[i] = distances[knns[i]]


    def point_type(expr: dict):
        if expr['is_val']:
            return 'val'
        elif expr['is_test']:
            return 'test'
        elif expr['is_neweq']:
            return 'neweq'
        return 'train'


    confusion_info = []
    pairs_seen = set()
    for i, k_dist in product(range(knns.shape[0]), range(knns.shape[1])):
        if eq_class_counts[expression_data[i]['eq_class']] < k_dist + 1:
            continue  # No other points should be NNs with this point
        j = knns[i, k_dist]
        pair = (i, j) if i < j else (j, i)
        if pair in pairs_seen:
            continue
        pairs_seen.add(pair)
        if expression_data[i]['eq_class'] == expression_data[j]['eq_class']:
            continue
        conf_str = (expression_data[i]['expr'], point_type(expression_data[i]),
                    eq_class_idx_to_names[expression_data[i]['eq_class']],
                    expression_data[j]['expr'], point_type(expression_data[j]),
                    eq_class_idx_to_names[expression_data[j]['eq_class']],
                    k_dist + 1, knn_distances[i, k_dist])
        confusion_info.append(conf_str)

    for info in sorted(confusion_info, key=lambda x: x[-1]):
        print("Expr %s (%s in class %s) and %s (%s in class %s) that are %s-NNs (%s)" % info)
