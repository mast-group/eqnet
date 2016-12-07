import sys
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist

from data.dataimport import import_data
from data.tree import Node
from encoders.baseencoder import AbstractEncoder


def tree_copy_with_start(tree: Node):
    if tree.name == 'Start':
        return tree

    root = Node('Start', ('child',), tree.symbol)

    original_root_copy = Node(tree.name, tree.properties, tree.symbol, parent=root)
    root.set_children_for_property('child', [original_root_copy])
    to_visit = [(tree, original_root_copy)]
    while len(to_visit) > 0:
        original, copy = to_visit.pop()
        for property_name in original.properties:
            children_copies = tuple(Node(c.name, c.properties, c.symbol, parent=copy) for c in original[property_name])
            copy.set_children_for_property(property_name, children_copies)
            to_visit.extend(list(zip(original[property_name], children_copies)))
    return root


def get_dataset_samples(filename: str):
    dataset_samples = []
    for name, code in import_data(filename).items():
        dataset_samples.append((''.join(code['original'][0]), code['original'][1]))
        for noisy_sample in code['noise']:
            dataset_samples.append((''.join(noisy_sample[0]), noisy_sample[1]))
    return set(dataset_samples)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage <encoderPkl> <dataset.json.gz> <testset.json.gz>")
        sys.exit(-1)

    testset_samples = get_dataset_samples(sys.argv[3])

    data = import_data(sys.argv[2])
    encoder = AbstractEncoder.load(sys.argv[1])

    expression_data, encodings = [], []
    eq_class_idx_to_names = {}
    eq_class_counts = defaultdict(int)


    def add_sample(data, eq_class_idx: int):
        sample_data = dict(
            tree=data[1],
            eq_class=eq_class_idx
        )
        expression_data.append(sample_data)

        representation = encoder.get_encoding(data)
        assert not np.isnan(np.sum(representation))
        encodings.append(representation)


    for eq_class_idx, (name, code) in enumerate(data.items()):
        eq_class_idx_to_names[eq_class_idx] = name
        eq_class_counts[name] = 1 + len(code['noise'])
        add_sample(code['original'], eq_class_idx)
        for noisy_sample in code['noise']:
            add_sample(noisy_sample, eq_class_idx)

    encodings = np.array(encodings)


    def get_knn_score_for(tree, k=5):
        tree = tree_copy_with_start(tree)
        tree_encoding = encoder.get_encoding([None, tree])  # This makes sure that token-based things fail
        tree_str_rep = str(tree)

        distances = cdist(np.atleast_2d(tree_encoding), encodings, 'cosine')
        knns = np.argsort(distances)[0]

        num_non_identical_nns = 0
        sum_equiv_nns = 0
        current_i = 0
        while num_non_identical_nns < k and current_i < len(knns) and eq_class_counts[
            tree.symbol] - 1 > num_non_identical_nns:
            expression_idx = knns[current_i]
            current_i += 1
            if eq_class_idx_to_names[expression_data[expression_idx]['eq_class']] == tree.symbol and str(
                    expression_data[expression_idx]['tree']) == tree_str_rep:
                continue  # This is an identical expression, move on
            num_non_identical_nns += 1
            if eq_class_idx_to_names[expression_data[expression_idx]['eq_class']] == tree.symbol:
                sum_equiv_nns += 1
        return "(%s-nn-stat: %s)" % (k, sum_equiv_nns / k)


    for i, (expr, test_sample) in enumerate(testset_samples):
        print(expr)
        print(test_sample.to_annotated_tree(annotator=get_knn_score_for))
        print('====================================================')
