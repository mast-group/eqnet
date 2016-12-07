import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data.dataimport import import_data


def plot_distribution(data, title):
    data = np.array([d for d in data])
    sns.distplot(data, rug=True)
    plt.title(title)
    plt.show()


if len(sys.argv) != 2:
    print("Usage <dataset.json.gz>")
    sys.exit(-1)

data = import_data(sys.argv[1]).values()


def num_noise_samples_per_original():
    for snippet in data:
        yield len(snippet["noise"])


plot_distribution(num_noise_samples_per_original(), title="Num noise samples per original")


def num_nodes_of_original():
    for snippet in data:
        original_tree = snippet["original"][1]
        yield sum(1 for node in original_tree)


plot_distribution(num_nodes_of_original(), title="Num Nodes in Original Nodes")


def num_missing_nodes_per_pair():
    for snippet in data:
        original_tree = snippet["original"][1]
        num_original_nodes = sum(1 for node in original_tree)
        for noise_sample in snippet["noise"]:
            num_noise_nodes = sum(1 for node in noise_sample[1])
            yield num_original_nodes - num_noise_nodes


print("Num 0 missing nodes per pair: %s" % sum(1 for k in num_missing_nodes_per_pair() if k == 0))
plot_distribution(num_missing_nodes_per_pair(), title="Num Removed Nodes per Original-Noise Pair")


def depth_of_tree():
    for snippet in data:
        original_tree = snippet["original"][1]
        yield original_tree.depth


plot_distribution(depth_of_tree(), title="Tree depth in original")


def diff_depth_per_pair():
    for snippet in data:
        original_tree = snippet["original"][1]
        original_depth = original_tree.depth
        for noise_sample in snippet["noise"]:
            noise_depth = noise_sample[1].depth
            yield original_depth - noise_depth


plot_distribution(diff_depth_per_pair(), title="Tree depth Diff in original vs noise")
