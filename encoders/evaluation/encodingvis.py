import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import TSNE

from data.dataimport import import_data
from encoders.baseencoder import AbstractEncoder

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage <encoderPkl> <dataset.json.gz> <testset.json.gz> <neweqtestset.json.gz>")
        sys.exit(-1)

    testset_samples = []
    for name, code in import_data(sys.argv[3]).items():
        testset_samples.append(''.join(code['original'][0]))
        for noisy_sample in code['noise']:
            testset_samples.append(''.join(noisy_sample[0]))
    testset_samples = set(testset_samples)

    neweq_test_set_eq_classes = set(import_data(sys.argv[4]).keys())

    data = import_data(sys.argv[2])
    encoder = AbstractEncoder.load(sys.argv[1])

    encodings, eq_classes_idxs, test_sample_idxs, neweq_samples_idxs = [], [], [], []
    eq_class_idx_to_names = {}

    for eq_class_idx, (name, code) in enumerate(data.items()):
        eq_class_idx_to_names[eq_class_idx] = name
        is_testset_example = ''.join(code['original'][0]) in testset_samples
        is_neweq_testset_example = name in neweq_test_set_eq_classes

        enc = encoder.get_encoding(code['original'])
        assert not np.isnan(np.sum(enc))
        encodings.append(enc)
        eq_classes_idxs.append(eq_class_idx)
        if is_testset_example:
            test_sample_idxs.append(len(encodings) - 1)
        elif is_neweq_testset_example:
            neweq_samples_idxs.append(len(encodings) - 1)

        for noisy_sample in code['noise']:
            is_testset_example = ''.join(noisy_sample[0]) in testset_samples
            enc = encoder.get_encoding(noisy_sample)
            assert not np.isnan(np.sum(enc))
            encodings.append(enc)
            eq_classes_idxs.append(eq_class_idx)
            if is_testset_example:
                test_sample_idxs.append(len(encodings) - 1)
            elif is_neweq_testset_example:
                neweq_samples_idxs.append(len(encodings) - 1)

    encodings, eq_classes_idxs = np.array(encodings), np.array(eq_classes_idxs)

    dists = squareform(pdist(encodings), 'cosine')

    print("Num points: %s" % encodings.shape[0])
    max_equivalence_class_idx = max(eq_class_idx_to_names)
    print("Num equivalence classes: %s" % (max_equivalence_class_idx + 1))
    tsne = TSNE(perplexity=50)
    vis_locs = tsne.fit_transform(dists)

    colormap = plt.get_cmap(matplotlib.rcParams['image.cmap'])
    for i in range(encodings.shape[0]):
        current_class_idx = eq_classes_idxs[i]
        for j in range(encodings.shape[0]):
            if i == j or current_class_idx != eq_classes_idxs[j]:
                continue
            color = colormap(int(float(current_class_idx) / max_equivalence_class_idx * colormap.N))
            color = list(color)
            color[-1] = .08
            color = tuple(color)
            plt.plot([vis_locs[i, 0], vis_locs[j, 0]], [vis_locs[i, 1], vis_locs[j, 1]], ':', color=color)

    plt.scatter(vis_locs[:, 0], vis_locs[:, 1], s=100, alpha=.4, c=eq_classes_idxs, zorder=9)

    for i in test_sample_idxs:
        plt.scatter(vis_locs[i, 0], vis_locs[i, 1], marker='^', zorder=10)
    for i in neweq_samples_idxs:
        plt.scatter(vis_locs[i, 0], vis_locs[i, 1], marker='*', zorder=10)

    plt.tight_layout()
    plt.show()
