import json
import os
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import TSNE

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

    validation_samples = get_dataset_samples(sys.argv[3])
    testset_samples = get_dataset_samples(sys.argv[4])
    newewset_samples = get_dataset_samples(sys.argv[5])

    data = import_data(sys.argv[2])
    encoder = AbstractEncoder.load(sys.argv[1])

    expression_data, encodings = [], []
    eq_class_idx_to_names = {}


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

        add_sample(code['original'], eq_class_idx)
        for noisy_sample in code['noise']:
            add_sample(noisy_sample, eq_class_idx)

    encodings = np.array(encodings)
    distances = squareform(pdist(encodings, 'cosine'))

    print("Num points: %s" % encodings.shape[0])
    max_equivalence_class_idx = max(eq_class_idx_to_names)
    print("Num equivalence classes: %s" % (max_equivalence_class_idx + 1))

    tsne = TSNE()
    vis_locs = tsne.fit_transform(distances)
    for i in range(encodings.shape[0]):
        expression_data[i]['xy'] = list(vis_locs[i])

    colormap = plt.get_cmap(matplotlib.rcParams['image.cmap'])
    eq_class_data = []
    for idx in range(len(eq_class_idx_to_names)):
        name = eq_class_idx_to_names[idx]
        color = colormap(int(float(idx) / max_equivalence_class_idx * colormap.N))
        eq_class_data.append({'name': name, 'color': color})

    with open(os.path.basename(sys.argv[2])[:-len('.json.gz')] + '-vis.json', 'w') as f:
        json.dump({'samples': expression_data, 'eq_classes': eq_class_data}, f)
